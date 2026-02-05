
import os
import logging
import time
import asyncio
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Lighter-Executor")

LIGHTER_API_KEY_INDEX = int(os.getenv("LIGHTER_API_KEY_INDEX", "3"))
LIGHTER_API_PRIVATE_KEY = os.getenv("LIGHTER_API_PRIVATE_KEY")
LIGHTER_ACCOUNT_INDEX = int(os.getenv("LIGHTER_ACCOUNT_INDEX", "0"))
LIGHTER_BASE_URL = os.getenv("LIGHTER_BASE_URL", "https://mainnet.zklighter.elliot.ai")

async def get_lighter_client():
    try:
        import lighter
        
        if not LIGHTER_API_PRIVATE_KEY:
            logger.error("Missing LIGHTER_API_PRIVATE_KEY in .env")
            return None
        
        logger.info(f"Initializing Lighter client (Account: {LIGHTER_ACCOUNT_INDEX}, API Key: {LIGHTER_API_KEY_INDEX})")
        client = lighter.SignerClient(
            url=LIGHTER_BASE_URL,
            api_private_keys={LIGHTER_API_KEY_INDEX: LIGHTER_API_PRIVATE_KEY},
            account_index=LIGHTER_ACCOUNT_INDEX
        )
        return client
        
    except ImportError as e:
        logger.error(f"lighter SDK not installed: {e}")
        logger.error("Install with: pip install git+https://github.com/elliottech/lighter-python.git")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Lighter: {e}")
        import traceback
        traceback.print_exc()
        return None

async def set_leverage_and_margin(client, market_id, leverage, use_isolated=True):
    """
    Set leverage and margin mode for a market.
    Note: There's a known bug in Lighter SDK where margin mode may default to cross.
    """
    try:
        margin_mode = 1 if use_isolated else 0
        
        logger.info(f"🔧 Setting Leverage: {leverage}x | Margin Mode: {'ISOLATED' if use_isolated else 'CROSS'}")
        
        tx, tx_hash, err = await client.update_leverage(
            market_index=market_id,
            leverage=leverage,
            margin_mode=margin_mode
        )
        
        if err:
            logger.warning(f"⚠️  Leverage update returned error: {err}")
            logger.warning("This might be due to a known SDK bug. Positions may use cross margin.")
            return False
        
        logger.info(f"✅ Leverage set to {leverage}x | TxHash: {tx_hash}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to set leverage: {e}")
        import traceback
        traceback.print_exc()
        return False

def round_size(size, decimals=4):
    return round(size, decimals)

def get_price_multiplier(price):
    if price >= 20000:
        return 10
    elif price >= 100:
        return 100
    elif price >= 1:
        return 10000
    else:
        return 1000000

def convert_price_to_int(price):
    multiplier = get_price_multiplier(price)
    return int(round(price * multiplier))


def round_price(price):
    if price >= 20000:
        return round(price, 1)
    elif price >= 1000:
        return round(price, 2)
    elif price >= 100:
        return round(price, 2)
    elif price >= 1:
        return round(price, 4)
    else:
        return round(price, 6)

async def execute_trade_sequence(market_id, direction, size, entry_price, sl_price, tps_list, tp_sizes_list, leverage):
    client = await get_lighter_client()
    if not client:
        logger.error("❌ Lighter client initialization failed")
        return False
    
    is_buy = (direction == 1)
    is_ask = not is_buy  # For Lighter: is_ask=True means SELL, is_ask=False means BUY
    entry_side = 'BUY' if is_buy else 'SELL'
    
    # Convert size to base_amount 
    # Size is already calculated with leverage in main.py: (INITIAL_CAPITAL * LEVERAGE) / price
    # Just convert to base_amount units: size * 100000
    # (1 base_amount = 0.00001 BTC or 10^-5)
    base_amount = int(size * 100000)
    
    if base_amount <= 0:
        logger.error("❌ Size is 0 after conversion")
        return False
    
    logger.info(f"🚀 Executing Market {market_id} {entry_side} | Size: {size:.6f} BTC | BaseAmount: {base_amount}")
    
    # Set leverage and margin mode (ISOLATED) before trading
    logger.info(f"📋 Attempting to set ISOLATED margin with {leverage}x leverage...")
    await set_leverage_and_margin(client, market_id, leverage, use_isolated=True)
    
    order_ids = {
        'entry': None,
        'stop_loss': None,
        'take_profits': []
    }
    
    client_order_base = int(time.time() * 1000) % 1000000

    
    try:
        # MARKET ENTRY
        try:
            logger.info(f"📊 Placing MARKET {entry_side} order...")
            
            # Slippage tolerance ONLY for market entry to ensure fill
            entry_slippage = 0.015 
            if is_ask: # SELL
                entry_limit_price = entry_price * (1 - entry_slippage)
            else: # BUY
                entry_limit_price = entry_price * (1 + entry_slippage)
                
            avg_price = convert_price_to_int(entry_limit_price)
            logger.info(f"📊 Placing MARKET {entry_side} order | Slippage: {entry_slippage*100}% | Worst Price: {entry_limit_price:.2f}")
            
            tx, tx_hash, err = await client.create_market_order(
                market_index=market_id,
                client_order_index=client_order_base + 1,
                base_amount=base_amount,
                avg_execution_price=avg_price,
                is_ask=is_ask,
                reduce_only=False
            )
            
            if err:
                logger.error(f"❌ Entry failed: {err}")
                return False
            
            order_ids['entry'] = client_order_base + 1
            logger.info(f"✅ Entry Order: {order_ids['entry']} | TxHash: {tx_hash}")
            
            logger.info("⏳ Waiting 2s for position update...")
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Entry Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # STOP LOSS
        try:
            logger.info(f"🛡️ Placing SL @ ${sl_price:.2f} | Size: {size:.5f} BTC")
            sl_price_int = convert_price_to_int(sl_price)
            
            # Add slippage to SL limit price to ensure fill when triggered
            sl_slippage = 0.015 # 1.5% slippage
            if is_ask:  # Entry was SELL, SL is BUY
                sl_limit_price = sl_price * (1 + sl_slippage)
            else:  # Entry was BUY, SL is SELL
                sl_limit_price = sl_price * (1 - sl_slippage)
            
            sl_limit_price_int = convert_price_to_int(sl_limit_price)
            logger.info(f"🛡️ SL Trigger: ${sl_price:.2f} | Limit (w/ slippage): ${sl_limit_price:.2f}")
            
            tx, tx_hash, err = await client.create_sl_order(
                market_index=market_id,
                client_order_index=client_order_base + 2,
                base_amount=base_amount,
                trigger_price=sl_price_int,
                price=sl_limit_price_int,  # Use limit price with slippage
                is_ask=not is_ask,
                reduce_only=True
            )
            
            if err:
                logger.warning(f"⚠ SL placement failed: {err}")
            else:
                order_ids['stop_loss'] = client_order_base + 2
                logger.info(f"✅ SL: {order_ids['stop_loss']} | Trigger: ${sl_price:.2f} | Limit: ${sl_limit_price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ SL Exception: {e}")
            import traceback
            traceback.print_exc()
        
        # TAKE PROFITS
        for i, (tp_price, pct) in enumerate(zip(tps_list, tp_sizes_list)):
            chunk_base_amount = int(base_amount * pct)
            if i == len(tps_list) - 1:
                allocated_so_far = sum([int(base_amount * p) for p in tp_sizes_list[:-1]])
                chunk_base_amount = base_amount - allocated_so_far
            
            if chunk_base_amount <= 0: continue
            
            try:
                tp_price_int = convert_price_to_int(tp_price)
                logger.info(f"🎯 Placing TP{i+1} @ ${tp_price:.2f} | BaseAmount: {chunk_base_amount}")
                
                tx, tx_hash, err = await client.create_tp_order(
                    market_index=market_id,
                    client_order_index=client_order_base + 3 + i,
                    base_amount=chunk_base_amount,
                    trigger_price=tp_price_int,
                    price=tp_price_int, # No slippage for TP
                    is_ask=not is_ask,
                    reduce_only=True
                )
                
                if err:
                    logger.warning(f"⚠ TP{i+1} placement failed: {err}")
                else:
                    order_ids['take_profits'].append({
                        'id': client_order_base + 3 + i,
                        'level': i + 1,
                        'price': tp_price
                    })
                    logger.info(f"✅ TP{i+1}: {client_order_base + 3 + i}")
            except Exception as e:
                logger.error(f"❌ TP{i+1} Exception: {e}")

    finally:
        await client.close()
    
    logger.info("=" * 60)
    logger.info("📋 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Entry: {order_ids['entry']}")
    logger.info(f"SL: {order_ids['stop_loss']}")
    logger.info(f"TPs: {len(order_ids['take_profits'])} orders")
    logger.info("=" * 60)
    
    return order_ids

async def cancel_order(market_id, order_index):
    client = await get_lighter_client()
    if not client:
        return False
    
    try:
        tx, tx_hash, err = await client.cancel_order(
            market_index=market_id,
            order_index=order_index
        )
        if err:
            logger.error(f"❌ Cancel failed: {err}")
            return False
        logger.info(f"✅ Cancelled: {order_index}")
        await client.close()
        return True
    except Exception as e:
        logger.error(f"❌ Cancel exception: {e}")
        return False

async def update_stop_loss(market_id, direction, new_sl_price, size, leverage, old_sl_order_index=None):
    client = await get_lighter_client()
    if not client:
        return False
    
    if old_sl_order_index:
        try:
            await client.cancel_order(
                market_index=market_id,
                order_index=old_sl_order_index
            )
            logger.info(f"✅ Old SL cancelled: {old_sl_order_index}")
        except Exception as e:
            logger.warning(f"⚠ Cancel old SL failed: {e}")
    
    is_buy = (direction == 1)
    entry_is_ask = not is_buy
    sl_side_is_ask = not entry_is_ask # SL is always opposite to Entry
    
    # Size passed is already leveraged
    base_amount = int(size * 100000)
    
    try:
        logger.info(f"🛡️ Updating SL to ${new_sl_price:.2f}...")
        sl_price_int = convert_price_to_int(new_sl_price)
        
        # Add slippage to SL limit price to ensure fill when triggered
        sl_slippage = 0.015  # 1.5% slippage
        if sl_side_is_ask:  # SL is SELL
            sl_limit_price = new_sl_price * (1 - sl_slippage)
        else:  # SL is BUY
            sl_limit_price = new_sl_price * (1 + sl_slippage)
        
        sl_limit_price_int = convert_price_to_int(sl_limit_price)
        logger.info(f"🛡️ SL Trigger: ${new_sl_price:.2f} | Limit (w/ slippage): ${sl_limit_price:.2f}")
        
        new_order_index = int(time.time() * 1000) % 1000000 + 2
        
        tx, tx_hash, err = await client.create_sl_order(
            market_index=market_id,
            client_order_index=new_order_index,
            base_amount=base_amount,
            trigger_price=sl_price_int,
            price=sl_limit_price_int,  # Use limit price with slippage
            is_ask=sl_side_is_ask,
            reduce_only=True
        )
        
        if err:
            logger.error(f"❌ New SL failed: {err}")
            return False
        
        logger.info(f"✅ New SL: {new_order_index} | Trigger: ${new_sl_price:.2f} | Limit: ${sl_limit_price:.2f}")
        return new_order_index
        
    except Exception as e:
        logger.error(f"❌ Update SL Exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await client.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python executor.py <MARKET_ID> <BUY|SELL> <SIZE> [LEVERAGE]")
        print("Example: python executor.py 1 BUY 0.01 50")
        sys.exit(1)
    
    market_id = int(sys.argv[1])
    dir_str = sys.argv[2].upper()
    size = float(sys.argv[3])
    leverage = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    direction = 1 if dir_str == "BUY" else -1
    
    print(f"🧪 TEST MODE: Market {market_id} {dir_str} Size: {size} Lev: {leverage}")
    
    print("⚠ Using mock price for testing")
    current_price = 91000.0
    
    ATR_MOCK = current_price * 0.01
    SL_DIST = ATR_MOCK * 1.2
    
    if direction == -1:
        sl_price = current_price + SL_DIST
        tps = [current_price - (SL_DIST * x) for x in [1.5, 2.5, 3.5, 5.0]]
    else:
        sl_price = current_price - SL_DIST
        tps = [current_price + (SL_DIST * x) for x in [1.5, 2.5, 3.5, 5.0]]
    
    tp_sizes = [0.3, 0.3, 0.2, 0.2]
    
    result = asyncio.run(execute_trade_sequence(market_id, direction, size, current_price, sl_price, tps, tp_sizes, leverage))
    
    if result:
        print("\n✅ Trade execution completed!")
        print(f"Order IDs: {result}")
    else:
        print("\n❌ Trade execution failed!")
