import os
import psycopg2
from psycopg2 import sql
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Establishes a connection to the database using the DATABASE_URL environment variable.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set.")
        return None
    
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def init_db():
    """
    Creates the necessary tables if they do not exist.
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS trade_logs (
        id SERIAL PRIMARY KEY,
        predicted_trade VARCHAR(50) NOT NULL,
        trade_date DATE NOT NULL,
        trade_time TIME NOT NULL,
        sl FLOAT NOT NULL,
        tps TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(create_table_query)
            
            create_placed_trades_query = """
            CREATE TABLE IF NOT EXISTS placed_trades (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                entry_price FLOAT NOT NULL,
                size FLOAT NOT NULL,
                leverage INT,
                entry_order_id VARCHAR(100),
                sl_price FLOAT,
                sl_order_id VARCHAR(100),
                tps_details TEXT,
                status VARCHAR(20) DEFAULT 'PLACED',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cur.execute(create_placed_trades_query)
            
            conn.commit()
            cur.close()
            logger.info("Database initialized successfully (trade_logs and placed_trades).")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
        finally:
            conn.close()

def insert_trade_log(predicted_trade, trade_date, trade_time, sl, tps):
    """
    Inserts a new trade log into the database.
    (Kept for backward compatibility and logging signals)
    """
    insert_query = """
    INSERT INTO trade_logs (predicted_trade, trade_date, trade_time, sl, tps)
    VALUES (%s, %s, %s, %s, %s);
    """
    
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            tps_str = json.dumps(tps) if isinstance(tps, list) else str(tps)
            
            cur.execute(insert_query, (predicted_trade, trade_date, trade_time, sl, tps_str))
            conn.commit()
            cur.close()
            logger.info(f"Trade signal logged: {predicted_trade} at {trade_time}")
        except Exception as e:
            logger.error(f"Error inserting trade log: {e}")
        finally:
            conn.close()

def insert_placed_trade(symbol, direction, entry_price, size, leverage, entry_order_id, sl_price, sl_order_id, tps_details, status="PLACED", error_message=None):
    """
    Inserts a record into the placed_trades table.
    
    :param tps_details: List of TP dicts or JSON string
    """
    insert_query = """
    INSERT INTO placed_trades (
        symbol, direction, entry_price, size, leverage, 
        entry_order_id, sl_price, sl_order_id, tps_details, 
        status, error_message
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            tps_str = json.dumps(tps_details) if isinstance(tps_details, (list, dict)) else str(tps_details)
            
            cur.execute(insert_query, (
                symbol, direction, entry_price, size, leverage,
                entry_order_id, sl_price, sl_order_id, tps_str,
                status, error_message
            ))
            conn.commit()
            cur.close()
            logger.info(f"Placed trade logged for {symbol} ({status})")
        except Exception as e:
            logger.error(f"Error inserting placed trade: {e}")
        finally:
            conn.close()
