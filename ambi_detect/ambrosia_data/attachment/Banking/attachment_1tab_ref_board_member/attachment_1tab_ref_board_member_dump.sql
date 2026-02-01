BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Balance REAL DEFAULT 0,
    Type TEXT ,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Accounts" VALUES(1,1,5000.0,'Checking');
INSERT INTO "Accounts" VALUES(2,2,3000.0,'Savings');
INSERT INTO "Accounts" VALUES(3,3,4000.0,'Checking');
INSERT INTO "Accounts" VALUES(4,4,2000.0,'Savings');
INSERT INTO "Accounts" VALUES(5,5,1000.0,'Checking');
CREATE TABLE Board_Members(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    MemberName TEXT,
    Role TEXT,
    OrganizationID INTEGER,
    FOREIGN KEY (OrganizationID) REFERENCES Organizations(ID));
INSERT INTO "Board_Members" VALUES(1,'John Smith','Independent Director',1);
INSERT INTO "Board_Members" VALUES(2,'Jane Doe','Independent Director',3);
INSERT INTO "Board_Members" VALUES(3,'Michael Johnson','Executive Manager',1);
INSERT INTO "Board_Members" VALUES(4,'Emily Brown','Executive Manager',2);
INSERT INTO "Board_Members" VALUES(5,'David Wilson','Non-executive Director',4);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATETIME,
    EmailAddress TEXT,
    PhoneNumber VARCHAR(20),
    AddressLine1 TEXT,
    AddressLine2 TEXT,
    City TEXT,
    State TEXT,
    ZipCode CHAR(5),
    Country TEXT);
INSERT INTO "Customers" VALUES(1,'Alice','Smith','1985-06-15','alice@example.com','+123456789','123 Main St.','','New York','NY','10001','USA');
INSERT INTO "Customers" VALUES(2,'Bob','Johnson','1990-02-22','bob@example.com','+098765432','456 Elm St.','','Los Angeles','CA','90001','USA');
INSERT INTO "Customers" VALUES(3,'Carol','Williams','1988-03-10','carol@example.com','+111222333','789 Oak St.','','Chicago','IL','60601','USA');
INSERT INTO "Customers" VALUES(4,'Daniel','Davis','1991-04-18','daniel@example.com','+44556677889','246 Pine St.','','Houston','TX','77001','USA');
INSERT INTO "Customers" VALUES(5,'Eva','Martin','1992-05-25','eva@example.com','+9998877665','321 Fir St.','','Phoenix','AZ','85001','USA');
CREATE TABLE Loans(
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Amount REAL,
    InterestRate REAL,
    StartDate DATE,
    EndDate DATE,
    Status TEXT ,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Loans" VALUES(1,1,15000.0,5.0,'2022-01-01','2024-12-31','Approved');
INSERT INTO "Loans" VALUES(2,2,10000.0,4.5,'2021-07-01','2023-06-30','Disbursed');
INSERT INTO "Loans" VALUES(3,3,20000.0,4.75,'2020-01-01','2022-12-31','Repaid');
INSERT INTO "Loans" VALUES(4,4,8000.0,5.25,'2021-04-01','2023-03-31','Pending');
INSERT INTO "Loans" VALUES(5,5,5000.0,4.75,'2021-10-01','2023-09-30','Denied');
CREATE TABLE Organizations(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT);
INSERT INTO "Organizations" VALUES(1,'Global Corp');
INSERT INTO "Organizations" VALUES(2,'Tech Inc.');
INSERT INTO "Organizations" VALUES(3,'Financial Group');
INSERT INTO "Organizations" VALUES(4,'Investment Holdings');
INSERT INTO "Organizations" VALUES(5,'Bank Solutions');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Organizations',5);
INSERT INTO "sqlite_sequence" VALUES('Board_Members',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Loans',5);
COMMIT;
