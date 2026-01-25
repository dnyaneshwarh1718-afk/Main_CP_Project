USE cap_prj;

-- 1) Standardize RAW datatypes --

-- ACCOUNT
ALTER TABLE account
MODIFY frequency VARCHAR(20);

-- LOAN
ALTER TABLE loan
MODIFY status VARCHAR(10);

-- DISP
ALTER TABLE disp
MODIFY type VARCHAR(20);

-- CARD
ALTER TABLE card
MODIFY type VARCHAR(20),
MODIFY issued VARCHAR(50);

-- ORDERS
ALTER TABLE orders
MODIFY k_symbol VARCHAR(30),
MODIFY amount DOUBLE;

-- TRANSACTION_DATA
ALTER TABLE transaction_data
MODIFY type VARCHAR(20),
MODIFY operation VARCHAR(50),
MODIFY k_symbol VARCHAR(30),
MODIFY bank VARCHAR(30),
MODIFY account VARCHAR(50);

-- CLIENT (birth_number sometimes should be VARCHAR, but keep INT if consistent)
ALTER TABLE client
MODIFY birth_number INT;


-- 2) Add GENERATED STORED Date columns -- 

DELIMITER $$

CREATE PROCEDURE sp_add_generated_date_columns()
BEGIN

    -- ACCOUNT: account_date
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'account'
          AND column_name = 'account_date'
    ) THEN
        ALTER TABLE account
        ADD COLUMN account_date DATE
        GENERATED ALWAYS AS (
            STR_TO_DATE(
                CONCAT(
                    CASE
                        WHEN LEFT(CAST(date AS CHAR), 2) >= '90'
                            THEN CONCAT('19', LEFT(CAST(date AS CHAR),2))
                        ELSE CONCAT('20', LEFT(CAST(date AS CHAR),2))
                    END,
                    MID(CAST(date AS CHAR), 3, 2),
                    RIGHT(CAST(date AS CHAR), 2)
                ),
                '%Y%m%d'
            )
        ) STORED;
    END IF;

    -- LOAN: loan_date
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'loan'
          AND column_name = 'loan_date'
    ) THEN
        ALTER TABLE loan
        ADD COLUMN loan_date DATE
        GENERATED ALWAYS AS (
            STR_TO_DATE(
                CONCAT(
                    CASE
                        WHEN LEFT(CAST(date AS CHAR), 2) >= '90'
                            THEN CONCAT('19', LEFT(CAST(date AS CHAR),2))
                        ELSE CONCAT('20', LEFT(CAST(date AS CHAR),2))
                    END,
                    MID(CAST(date AS CHAR), 3, 2),
                    RIGHT(CAST(date AS CHAR), 2)
                ),
                '%Y%m%d'
            )
        ) STORED;
    END IF;

    -- TRANSACTION_DATA: txn_date
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'transaction_data'
          AND column_name = 'txn_date'
    ) THEN
        ALTER TABLE transaction_data
        ADD COLUMN txn_date DATE
        GENERATED ALWAYS AS (
            STR_TO_DATE(
                CONCAT(
                    CASE
                        WHEN LEFT(CAST(date AS CHAR), 2) >= '90'
                            THEN CONCAT('19', LEFT(CAST(date AS CHAR),2))
                        ELSE CONCAT('20', LEFT(CAST(date AS CHAR),2))
                    END,
                    MID(CAST(date AS CHAR), 3, 2),
                    RIGHT(CAST(date AS CHAR), 2)
                ),
                '%Y%m%d'
            )
        ) STORED;
    END IF;

    -- CARD: issued_date
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'card'
          AND column_name = 'issued_date'
    ) THEN
        ALTER TABLE card
        ADD COLUMN issued_date DATE
        GENERATED ALWAYS AS (
            STR_TO_DATE(
                CONCAT(
                    CASE 
                        WHEN LEFT(SUBSTRING_INDEX(CAST(issued AS CHAR), ' ', 1), 2) >= '90'
                            THEN CONCAT('19', LEFT(SUBSTRING_INDEX(CAST(issued AS CHAR), ' ', 1), 2))
                        ELSE CONCAT('20', LEFT(SUBSTRING_INDEX(CAST(issued AS CHAR), ' ', 1), 2))
                    END,
                    MID(SUBSTRING_INDEX(CAST(issued AS CHAR), ' ', 1), 3, 2),
                    RIGHT(SUBSTRING_INDEX(CAST(issued AS CHAR), ' ', 1), 2)
                ),
                '%Y%m%d'
            )
        ) STORED;
    END IF;

END$$

DELIMITER ;

CALL sp_add_generated_date_columns();

DROP PROCEDURE sp_add_generated_date_columns;


-- 3) Create Indexes on RAW tables --


DELIMITER $$

CREATE PROCEDURE sp_create_all_indexes_safe()
BEGIN

    -- ACCOUNT
  
    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'account'
          AND index_name = 'idx_account_district_id'
    ) THEN DROP INDEX idx_account_district_id ON account; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'account'
          AND index_name = 'idx_account_date'
    ) THEN DROP INDEX idx_account_date ON account; END IF;

    CREATE INDEX idx_account_district_id ON account(district_id);
    CREATE INDEX idx_account_date ON account(account_date);


    -- LOAN
    
    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'loan'
          AND index_name = 'idx_loan_account_id'
    ) THEN DROP INDEX idx_loan_account_id ON loan; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'loan'
          AND index_name = 'idx_loan_date'
    ) THEN DROP INDEX idx_loan_date ON loan; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'loan'
          AND index_name = 'idx_loan_status'
    ) THEN DROP INDEX idx_loan_status ON loan; END IF;

    CREATE INDEX idx_loan_account_id ON loan(account_id);
    CREATE INDEX idx_loan_date ON loan(loan_date);
    CREATE INDEX idx_loan_status ON loan(status);

  
    -- DISP
  
    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'disp'
          AND index_name = 'idx_disp_account_id'
    ) THEN DROP INDEX idx_disp_account_id ON disp; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'disp'
          AND index_name = 'idx_disp_client_id'
    ) THEN DROP INDEX idx_disp_client_id ON disp; END IF;

    CREATE INDEX idx_disp_account_id ON disp(account_id);
    CREATE INDEX idx_disp_client_id ON disp(client_id);

   
    -- CARD
 
    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'card'
          AND index_name = 'idx_card_disp_id'
    ) THEN DROP INDEX idx_card_disp_id ON card; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'card'
          AND index_name = 'idx_card_issued_date'
    ) THEN DROP INDEX idx_card_issued_date ON card; END IF;

    CREATE INDEX idx_card_disp_id ON card(disp_id);
    CREATE INDEX idx_card_issued_date ON card(issued_date);

  
    -- ORDERS
  
    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'orders'
          AND index_name = 'idx_orders_account_id'
    ) THEN DROP INDEX idx_orders_account_id ON orders; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'orders'
          AND index_name = 'idx_orders_k_symbol'
    ) THEN DROP INDEX idx_orders_k_symbol ON orders; END IF;

    CREATE INDEX idx_orders_account_id ON orders(account_id);
    CREATE INDEX idx_orders_k_symbol ON orders(k_symbol);

  
    -- TRANSACTION_DATA
  
    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'transaction_data'
          AND index_name = 'idx_txn_account_id'
    ) THEN DROP INDEX idx_txn_account_id ON transaction_data; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'transaction_data'
          AND index_name = 'idx_txn_account_date'
    ) THEN DROP INDEX idx_txn_account_date ON transaction_data; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'transaction_data'
          AND index_name = 'idx_txn_k_symbol'
    ) THEN DROP INDEX idx_txn_k_symbol ON transaction_data; END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = 'transaction_data'
          AND index_name = 'idx_txn_operation'
    ) THEN DROP INDEX idx_txn_operation ON transaction_data; END IF;

    CREATE INDEX idx_txn_account_id ON transaction_data(account_id);
    CREATE INDEX idx_txn_account_date ON transaction_data(account_id, txn_date);
    CREATE INDEX idx_txn_k_symbol ON transaction_data(k_symbol);
    CREATE INDEX idx_txn_operation ON transaction_data(operation);

END$$

DELIMITER ;

CALL sp_create_all_indexes_safe();
DROP PROCEDURE sp_create_all_indexes_safe;


