USE cap_prj;

-- Master Table As Per Problem Statement
-- Grain: 1 Row Per Loan_ID
-- Target Column: Status

DROP TABLE IF EXISTS loan_master;

CREATE TABLE loan_master AS
SELECT
    -- Loan (Target Base)
    ln.loan_id     AS Loan_ID,
    ln.account_id  AS Account_ID,
    ln.loan_date   AS Loan_Date,
    ln.loan_amount AS Loan_Amount,
    ln.duration    AS Duration,
    ln.payments    AS Payments,
    ln.status      AS Status,

    -- Account Details
    acc.frequency    AS Frequency,
    acc.account_date AS Account_Date,

    -- Disposition (Aggregates)
    COALESCE(da.primary_client_id, 0) AS Client_ID,
    COALESCE(da.owner_count, 0)       AS Owner_Count,
    COALESCE(da.user_count, 0)        AS User_Count,
    COALESCE(da.is_joint_account, 0)  AS Is_Joint_Account,

    -- Transaction Behavior (Aggregates)
    COALESCE(ta.txn_count, 0)          AS Txn_Count,
    COALESCE(ta.active_days, 0)        AS Active_Days,
    COALESCE(ta.avg_trans_amount, 0)   AS Avg_Trans_Amount,
    COALESCE(ta.total_trans_amount, 0) AS Total_Trans_Amount,
    COALESCE(ta.avg_balance, 0)        AS Avg_Balance,
    COALESCE(ta.balance_volatility, 0) AS Balance_Volatility,
    COALESCE(ta.total_credit, 0)       AS Total_Credit,
    COALESCE(ta.total_debit, 0)        AS Total_Debit,
    COALESCE(ta.net_cashflow, 0)       AS Net_Cashflow,

    -- Orders (Aggregates)
    COALESCE(oa.order_count, 0)        AS Order_Count,
    COALESCE(oa.total_order_amount, 0) AS Total_Order_Amount,
    COALESCE(oa.avg_order_amount, 0)   AS Avg_Order_Amount,

    -- Cards (Aggregates)
    COALESCE(ca.card_count, 0)      AS Card_Count,
    COALESCE(ca.has_card, 0)        AS Has_Card,
    COALESCE(ca.gold_card_count, 0) AS Gold_Card_Count,

    -- District Info (via Account → District)
    dist.a2  AS District_Name,
    dist.a3  AS Region,
    dist.a11 AS Avg_Salary,
    dist.a12 AS Unemployment_Rate_1,
    dist.a13 AS Unemployment_Rate_2,
    (dist.a12 + dist.a13) / 2 AS Avg_Unemployment_Rate,
    dist.a14 AS Entrepreneurs_Per_1000,
    dist.a15 AS Crimes_1,
    dist.a16 AS Crimes_2

FROM vw_clean_loan ln
LEFT JOIN vw_clean_account acc
    ON ln.account_id = acc.account_id
LEFT JOIN txn_agg ta
    ON ln.account_id = ta.account_id
LEFT JOIN order_agg oa
    ON ln.account_id = oa.account_id
LEFT JOIN card_agg ca
    ON ln.account_id = ca.account_id
LEFT JOIN disp_agg da
    ON ln.account_id = da.account_id
LEFT JOIN district dist
    ON acc.district_id = dist.a1;


-- Main SQL Query
SELECT *
FROM loan_master;
