BEGIN TRANSACTION;
CREATE TABLE Company (
    company_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    location TEXT
);
INSERT INTO "Company" VALUES(1,'ABC Corporation','New York');
INSERT INTO "Company" VALUES(2,'XYZ Inc.','California');
INSERT INTO "Company" VALUES(3,'LNM Limited','Texas');
INSERT INTO "Company" VALUES(4,'MNO Enterprises','Florida');
INSERT INTO "Company" VALUES(5,'PQR Holdings','Illinois');
CREATE TABLE Financial_Analysts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INT
);
INSERT INTO "Financial_Analysts" VALUES(1,'John Doe',3);
INSERT INTO "Financial_Analysts" VALUES(2,'Jane Smith',2);
INSERT INTO "Financial_Analysts" VALUES(3,'Alice Johnson',10);
INSERT INTO "Financial_Analysts" VALUES(4,'Bob Brown',7);
INSERT INTO "Financial_Analysts" VALUES(5,'Emma Wilson',6);
CREATE TABLE Financial_Analysts_JobPostings (
    financial_analyst_id INTEGER,
    job_posting_id INTEGER,
    FOREIGN KEY(financial_analyst_id) REFERENCES Financial_Analysts(id),
    FOREIGN KEY(job_posting_id) REFERENCES JobPostings(job_posting_id)
);
CREATE TABLE Financial_Analysts_Metrics (
    financial_analyst_id INTEGER,
    metric_id INTEGER,
    FOREIGN KEY(financial_analyst_id) REFERENCES Financial_Analysts(id),
    FOREIGN KEY(metric_id) REFERENCES Metrics(metric_id)
);
INSERT INTO "Financial_Analysts_Metrics" VALUES(1,1);
INSERT INTO "Financial_Analysts_Metrics" VALUES(2,1);
INSERT INTO "Financial_Analysts_Metrics" VALUES(3,1);
INSERT INTO "Financial_Analysts_Metrics" VALUES(4,1);
INSERT INTO "Financial_Analysts_Metrics" VALUES(5,1);
INSERT INTO "Financial_Analysts_Metrics" VALUES(5,4);
CREATE TABLE JobPostings (
    job_posting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    salary REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "JobPostings" VALUES(1,'Financial Analyst','Analyze market trends and make recommendations based on research',75000.0,'2024-01-30 19:22:37');
INSERT INTO "JobPostings" VALUES(2,'Senior Financial Analyst','Lead a team of analysts in providing investment advice',100000.0,'2024-01-30 19:22:37');
INSERT INTO "JobPostings" VALUES(3,'Associate Financial Analyst','Assist senior analysts in preparing reports and forecasts',60000.0,'2024-01-30 19:22:37');
INSERT INTO "JobPostings" VALUES(4,'Financial Modelling Analyst','Develop financial models to assist in decision-making processes',70000.0,'2024-01-30 19:22:37');
INSERT INTO "JobPostings" VALUES(5,'Equity Research Analyst','Conduct analysis on specific industries and companies to provide buy/sell/hold recommendations',80000.0,'2024-01-30 19:22:37');
CREATE TABLE JobPostings_Company (
    job_posting_id INTEGER,
    company_id INTEGER,
    FOREIGN KEY(job_posting_id) REFERENCES JobPostings(job_posting_id),
    FOREIGN KEY(company_id) REFERENCES Company(company_id)
);
CREATE TABLE Metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT);
INSERT INTO "Metrics" VALUES(1,'Earnings_Per_Share');
INSERT INTO "Metrics" VALUES(2,'Revenue');
INSERT INTO "Metrics" VALUES(3,'Gross_Profit');
INSERT INTO "Metrics" VALUES(4,'Net_Income');
INSERT INTO "Metrics" VALUES(5,'Market_Capitalization');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Financial_Analysts',5);
INSERT INTO "sqlite_sequence" VALUES('Metrics',5);
INSERT INTO "sqlite_sequence" VALUES('Company',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',5);
COMMIT;
