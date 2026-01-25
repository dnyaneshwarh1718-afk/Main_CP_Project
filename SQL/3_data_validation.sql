-- 1) Data Validation
-- Table Row Counts

SELECT table_name, row_count
FROM (
    SELECT 'account' AS table_name, COUNT(*) AS row_count FROM account
    UNION ALL SELECT 'card', COUNT(*) FROM card
    UNION ALL SELECT 'client', COUNT(*) FROM client
    UNION ALL SELECT 'disp', COUNT(*) FROM disp
    UNION ALL SELECT 'district', COUNT(*) FROM district
    UNION ALL SELECT 'loan', COUNT(*) FROM loan
    UNION ALL SELECT 'orders', COUNT(*) FROM orders
    UNION ALL SELECT 'transaction_data', COUNT(*) FROM transaction_data
) t
ORDER BY row_count DESC;

-- 2) primary_key_checks

-- loan_id should be unique
SELECT loan_id, COUNT(*) AS id_count
FROM loan
GROUP BY loan_id
HAVING COUNT(*) > 1;

-- account_id should be unique
SELECT account_id, COUNT(*) AS id_count
FROM account
GROUP BY account_id
HAVING COUNT(*) > 1;

-- _id should be unique
SELECT trans_id, COUNT(*) AS id_count
FROM transaction_data
GROUP BY trans_id
HAVING COUNT(*) > 1;

-- 3) foreign_key_checks
-- orphan loans
select count(*) as orphan_loans
from loan l
left join account a
on l.account_id = a.account_id
where a.account_id is null;

-- orphan transactions
select count(*) as orphan_transactions
from transaction_data t
left join account a
on t.account_id = a.account_id
where a.account_id is null;

-- orphan orders
select count(*) as orphan_orders
from orders o
left join account a
on o.account_id = a.account_id
where a.account_id is null;

-- Account without district
select count(*) as Account_missing_district
from account a
left join  district d
on a.district_id = d.A1
where d.A1 is null;

-- 4) null_checks
-- Null checks in loan table
select 
	sum(loan_id is null) as null_loan_id,
    sum(account_id is null) as null_account_id,
    sum(date is null) as date_is_null,
    sum(amount is null) as null_amount,
    sum(duration is null) as null_duration,
    sum(payments is null) as null_payments,
    sum(status is null) as null_status
from loan;

-- Null checks in account table
select 
	sum(account_id   is null) as null_account_id,
    sum(district_id  is null) as null_district_id,
    sum(date is null) as date_is_null,
    sum(frequency is null) as null_frequency
from account;

-- 5) Domain + Business Rule Checks (Loan)

SELECT status, COUNT(*) AS cnt
FROM loan
GROUP BY status
ORDER BY cnt DESC;

SELECT
    SUM(amount <= 0 OR amount IS NULL) AS bad_amount,
    SUM(duration <= 0 OR duration IS NULL) AS bad_duration,
    SUM(payments <= 0 OR payments IS NULL) AS bad_payments
FROM loan;


