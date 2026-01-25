-- 1) transaction_Aggregation

DROP TABLE IF EXISTS txn_agg;

CREATE TABLE txn_agg AS
SELECT
    account_id,
    COUNT(*) AS txn_count,
    COUNT(DISTINCT txn_date) AS active_days,
    ROUND(AVG(amount), 2) AS avg_trans_amount,
    ROUND(SUM(amount), 2) AS total_trans_amount,
    ROUND(AVG(balance), 2) AS avg_balance,
    ROUND(STDDEV_POP(balance), 2) AS balance_volatility,
	min(Balance) as Min_Balance,
    Max(Balance) As Max_Balance,
    ROUND(SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), 2) AS total_credit,
    ROUND(SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END), 2) AS total_debit,
    ROUND(
        SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) -
        SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END),
    2) AS net_cashflow,
    sum(case when k_symbol is null then 1 else 0 end) as K_Symbol_Null_Count,
    sum(case when k_symbol is not null then 1 else 0 end) as K_Symbol_known_Count
FROM vw_clean_transaction_data
GROUP BY account_id;

-- 2) orders_Aggregation
DROP TABLE IF EXISTS order_agg;
CREATE TABLE order_agg AS
SELECT
    account_id,
    COUNT(*) AS order_count,
    ROUND(SUM(amount), 2) AS total_order_amount,
    ROUND(AVG(amount), 2) AS avg_order_amount,
	min(amount) as Min_Order_amount,
    max(Amount) as Max_Order_Amount
FROM vw_clean_orders
GROUP BY account_id;

-- 3) disp_Aggregation
DROP TABLE IF EXISTS disp_agg;
CREATE TABLE disp_agg AS
SELECT
    Account_ID,
    COUNT(*) AS total_clients_on_account,
    SUM(CASE WHEN disp_type = 'OWNER' THEN 1 ELSE 0 END) AS owner_count,
    SUM(CASE WHEN disp_type = 'USER' THEN 1 ELSE 0 END) AS user_count,
    CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END AS is_joint_account,

    -- for master table (one client_id per account rule)
    MAX(CASE WHEN disp_type = 'OWNER' THEN Client_ID END) AS primary_client_id
FROM vw_clean_disp
GROUP BY Account_ID;

-- 4) card_Aggregation
DROP TABLE IF EXISTS card_agg;

CREATE TABLE card_agg AS
SELECT
    d.Account_ID,
    COUNT(c.Card_ID) AS Card_Count,
    CASE WHEN COUNT(c.Card_ID) > 0 THEN 1 ELSE 0 END AS Has_Card,
    SUM(CASE WHEN c.card_type='GOLD' THEN 1 ELSE 0 END) AS Gold_Card_Count,
    SUM(CASE WHEN c.card_type='JUNIOR' THEN 1 ELSE 0 END) AS Junior_Card_Count,
    SUM(CASE WHEN c.card_type='CLASSIC' THEN 1 ELSE 0 END) AS Classic_Card_Count
FROM vw_clean_disp d
LEFT JOIN vw_clean_card c
ON d.Disp_ID = c.Disp_ID
GROUP BY d.Account_ID;


