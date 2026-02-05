// @generated automatically by Diesel CLI.

diesel::table! {
    users (id) {
        id -> Uuid,
        #[max_length = 255]
        username -> Varchar,
        #[max_length = 255]
        api_key -> Varchar,
        account_index -> Int4,
        api_key_index -> Int4,
        created_at -> Timestamp,
        updated_at -> Timestamp,
    }
}
