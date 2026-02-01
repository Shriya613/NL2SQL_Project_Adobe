BEGIN TRANSACTION;
CREATE TABLE Housing_Finance_Agencies (
    agency_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    rating VARCHAR(10) 
);
INSERT INTO "Housing_Finance_Agencies" VALUES(1,'Fannie Mae','AAA');
INSERT INTO "Housing_Finance_Agencies" VALUES(2,'Freddie Mac','AAA');
INSERT INTO "Housing_Finance_Agencies" VALUES(3,'Ginnie Mae','AA+');
INSERT INTO "Housing_Finance_Agencies" VALUES(4,'Quicken Loans','A+');
INSERT INTO "Housing_Finance_Agencies" VALUES(5,'Wells Fargo Home Mortgage','A+');
CREATE TABLE Housing_Finance_Agencies_Loan_Programs (
    housing_finance_agency_id INTEGER,
    loan_program_id INTEGER,
    FOREIGN KEY (housing_finance_agency_id) REFERENCES Housing_Finance_Agencies (agency_id),
    FOREIGN KEY (loan_program_id) REFERENCES Loan_Programs (program_id),
    PRIMARY KEY (housing_finance_agency_id, loan_program_id)
);
INSERT INTO "Housing_Finance_Agencies_Loan_Programs" VALUES(1,4);
INSERT INTO "Housing_Finance_Agencies_Loan_Programs" VALUES(2,4);
INSERT INTO "Housing_Finance_Agencies_Loan_Programs" VALUES(3,4);
INSERT INTO "Housing_Finance_Agencies_Loan_Programs" VALUES(4,4);
INSERT INTO "Housing_Finance_Agencies_Loan_Programs" VALUES(5,4);
INSERT INTO "Housing_Finance_Agencies_Loan_Programs" VALUES(5,1);
CREATE TABLE Housing_Market (
    id INTEGER PRIMARY KEY,
    city VARCHAR(255),
    median_home_price REAL
);
INSERT INTO "Housing_Market" VALUES(1,'New York',650000.0);
INSERT INTO "Housing_Market" VALUES(2,'Los Angeles',780000.0);
INSERT INTO "Housing_Market" VALUES(3,'Chicago',390000.0);
INSERT INTO "Housing_Market" VALUES(4,'Miami',550000.0);
INSERT INTO "Housing_Market" VALUES(5,'San Francisco',1200000.0);
CREATE TABLE Loan_Programs (
    program_id INTEGER PRIMARY KEY,
    description TEXT,
    interest_rate REAL
);
INSERT INTO "Loan_Programs" VALUES(1,'Conventional Loan',3.5);
INSERT INTO "Loan_Programs" VALUES(2,'Adjustable Rate Mortgage',3.25);
INSERT INTO "Loan_Programs" VALUES(3,'Fixed-Rate Mortgage',3.75);
INSERT INTO "Loan_Programs" VALUES(4,'First-Time Home Buyer Program',3.25);
INSERT INTO "Loan_Programs" VALUES(5,'VA Loan',2.75);
CREATE TABLE RealEstateAgents (
    agent_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    years_experience INTEGER 
);
INSERT INTO "RealEstateAgents" VALUES(1,'John Doe',10);
INSERT INTO "RealEstateAgents" VALUES(2,'Jane Smith',5);
INSERT INTO "RealEstateAgents" VALUES(3,'Michael Brown',15);
INSERT INTO "RealEstateAgents" VALUES(4,'Emily Johnson',7);
INSERT INTO "RealEstateAgents" VALUES(5,'William Davis',20);
COMMIT;
