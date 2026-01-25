USE cap_prj;

-- CLEAN VIEW: ACCOUNT
DROP VIEW IF EXISTS vw_clean_account;
CREATE VIEW vw_clean_account AS
SELECT
    account_id,
    district_id,
    TRIM(frequency) AS frequency,
    account_date
FROM account;

-- CLEAN VIEW: LOAN
DROP VIEW IF EXISTS vw_clean_loan;
CREATE VIEW vw_clean_loan AS
SELECT
    loan_id,
    account_id,
    loan_date,
    amount AS loan_amount,
    duration,
    payments,
    TRIM(status) AS status
FROM loan;

-- CLEAN VIEW: TRANSACTION_DATA
DROP VIEW IF EXISTS vw_clean_transaction_data;
CREATE VIEW vw_clean_transaction_data AS
SELECT
    trans_id,
    account_id,
    txn_date,
    COALESCE(amount, 0) AS amount,
    COALESCE(balance, 0) AS balance,
    TRIM(type) AS txn_type,
    TRIM(operation) AS operation,
    TRIM(k_symbol) AS k_symbol,
    TRIM(bank) AS bank,
    TRIM(account) AS bank_account
FROM transaction_data;

-- CLEAN VIEW: ORDERS
DROP VIEW IF EXISTS vw_clean_orders;
CREATE VIEW vw_clean_orders AS
SELECT
    order_id,
    account_id,
    TRIM(k_symbol) AS k_symbol,
    COALESCE(amount, 0) AS amount
FROM orders;

-- CLEAN VIEW: CARD
DROP VIEW IF EXISTS vw_clean_card;
CREATE VIEW vw_clean_card AS
SELECT
    card_id,
    disp_id,
    TRIM(type) AS card_type,
    issued_date
FROM card;

-- CLEAN VIEW: DISP
DROP VIEW IF EXISTS vw_clean_disp;
CREATE VIEW vw_clean_disp AS
SELECT
    disp_id,
    client_id,
    account_id,
    TRIM(type) AS disp_type
FROM disp;

-- CLEAN VIEW: CLIENT
DROP VIEW IF EXISTS vw_clean_client;
CREATE VIEW vw_clean_client AS
SELECT
    client_id,
    birth_number,
    district_id
FROM client;

-- CLEAN VIEW: DISTRICT
DROP VIEW IF EXISTS vw_clean_district;

CREATE VIEW vw_clean_district AS
SELECT
    A1 as district_id,
    TRIM(A2) AS district_name,
    TRIM(A3) AS region,
    TRIM(A4) AS no_of_inhabitants,
    A5,
    A6,
    A7,
    A8,
    A9,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
    A16
FROM district;
