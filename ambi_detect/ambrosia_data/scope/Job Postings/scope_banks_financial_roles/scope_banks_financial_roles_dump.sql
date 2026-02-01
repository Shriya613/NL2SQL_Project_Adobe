BEGIN TRANSACTION;
CREATE TABLE Banks(
    BankID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Banks" VALUES(1,'Bank A','New York');
INSERT INTO "Banks" VALUES(2,'Bank B','Los Angeles');
INSERT INTO "Banks" VALUES(3,'Bank C','Chicago');
INSERT INTO "Banks" VALUES(4,'Bank D','Miami');
INSERT INTO "Banks" VALUES(5,'Bank E','Houston');
CREATE TABLE Banks_Financial_Roles(
    BankID INTEGER,
    RoleID INTEGER,
    FOREIGN KEY (BankID) REFERENCES Banks(BankID),
    FOREIGN KEY (RoleID) REFERENCES Financial_Roles(RoleID),
    PRIMARY KEY (BankID, RoleID)
);
INSERT INTO "Banks_Financial_Roles" VALUES(1,1);
INSERT INTO "Banks_Financial_Roles" VALUES(2,1);
INSERT INTO "Banks_Financial_Roles" VALUES(3,1);
INSERT INTO "Banks_Financial_Roles" VALUES(4,1);
INSERT INTO "Banks_Financial_Roles" VALUES(5,1);
INSERT INTO "Banks_Financial_Roles" VALUES(5,2);
CREATE TABLE Financial_Roles(
    RoleID INTEGER PRIMARY KEY AUTOINCREMENT,
    RoleName VARCHAR(255)
);
INSERT INTO "Financial_Roles" VALUES(1,'Financial Analyst');
INSERT INTO "Financial_Roles" VALUES(2,'Financial Advisor');
INSERT INTO "Financial_Roles" VALUES(3,'Investment Manager');
INSERT INTO "Financial_Roles" VALUES(4,'Portfolio Manager');
INSERT INTO "Financial_Roles" VALUES(5,'Risk Analyst');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Financial_Roles',5);
COMMIT;
