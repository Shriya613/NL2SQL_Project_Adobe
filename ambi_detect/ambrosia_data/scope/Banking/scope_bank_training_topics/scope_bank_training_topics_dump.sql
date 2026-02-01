BEGIN TRANSACTION;
CREATE TABLE Bank_Training (
    training_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_id INTEGER,
    employee_id INTEGER,
    FOREIGN KEY(bank_id) REFERENCES Banks(id),
    FOREIGN KEY(employee_id) REFERENCES Employees(id)
);
INSERT INTO "Bank_Training" VALUES(1,1,1);
INSERT INTO "Bank_Training" VALUES(2,2,2);
INSERT INTO "Bank_Training" VALUES(3,3,3);
INSERT INTO "Bank_Training" VALUES(4,4,4);
INSERT INTO "Bank_Training" VALUES(5,5,5);
CREATE TABLE Bank_Training_Topics (
    bank_training_id INTEGER,
    topic_id INTEGER,
    FOREIGN KEY(bank_training_id) REFERENCES Bank_Training(training_id),
    FOREIGN KEY(topic_id) REFERENCES Topics(id),
    PRIMARY KEY(bank_training_id, topic_id)
);
INSERT INTO "Bank_Training_Topics" VALUES(1,1);
INSERT INTO "Bank_Training_Topics" VALUES(1,2);
INSERT INTO "Bank_Training_Topics" VALUES(2,1);
INSERT INTO "Bank_Training_Topics" VALUES(2,2);
INSERT INTO "Bank_Training_Topics" VALUES(3,1);
INSERT INTO "Bank_Training_Topics" VALUES(4,1);
INSERT INTO "Bank_Training_Topics" VALUES(5,1);
CREATE TABLE Banks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Banks" VALUES(1,'ABC Bank','New York');
INSERT INTO "Banks" VALUES(2,'XYZ Bank','California');
INSERT INTO "Banks" VALUES(3,'LMN Bank','Texas');
INSERT INTO "Banks" VALUES(4,'PQR Bank','Florida');
INSERT INTO "Banks" VALUES(5,'STU Bank','Illinois');
CREATE TABLE Employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    bank_id INTEGER,
    FOREIGN KEY(bank_id) REFERENCES Banks(id)
);
INSERT INTO "Employees" VALUES(1,'John','Doe',1);
INSERT INTO "Employees" VALUES(2,'Jane','Smith',2);
INSERT INTO "Employees" VALUES(3,'Mike','Brown',3);
INSERT INTO "Employees" VALUES(4,'Emma','White',4);
INSERT INTO "Employees" VALUES(5,'Tom','Black',5);
CREATE TABLE Topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value TEXT
);
INSERT INTO "Topics" VALUES(1,'Customer Service Skills');
INSERT INTO "Topics" VALUES(2,'Fraud Prevention');
INSERT INTO "Topics" VALUES(3,'Interest Rate Management');
INSERT INTO "Topics" VALUES(4,'Loan Processing');
INSERT INTO "Topics" VALUES(5,'Investment Strategies');
CREATE TABLE TrainingSessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATETIME,
    trainer_id INTEGER,
    topic_id INTEGER,
    FOREIGN KEY(trainer_id) REFERENCES Employees(id),
    FOREIGN KEY(topic_id) REFERENCES Topics(id)
);
INSERT INTO "TrainingSessions" VALUES(1,'2021-10-01 09:00:00',1,1);
INSERT INTO "TrainingSessions" VALUES(2,'2021-10-08 09:00:00',2,2);
INSERT INTO "TrainingSessions" VALUES(3,'2021-10-15 09:00:00',3,1);
INSERT INTO "TrainingSessions" VALUES(4,'2021-10-22 09:00:00',4,2);
INSERT INTO "TrainingSessions" VALUES(5,'2021-10-29 09:00:00',5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Employees',5);
INSERT INTO "sqlite_sequence" VALUES('TrainingSessions',5);
INSERT INTO "sqlite_sequence" VALUES('Topics',5);
INSERT INTO "sqlite_sequence" VALUES('Bank_Training',5);
COMMIT;
