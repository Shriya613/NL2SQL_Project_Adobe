BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountId INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerId INTEGER,
    Type TEXT,
    Balance REAL ,
    FOREIGN KEY(CustomerId) REFERENCES Customers(CustomerId));
INSERT INTO "Accounts" VALUES(1,1,'Checking',1000.0);
INSERT INTO "Accounts" VALUES(2,1,'Savings',5000.0);
INSERT INTO "Accounts" VALUES(3,2,'Checking',2000.0);
INSERT INTO "Accounts" VALUES(4,2,'Savings',3000.0);
INSERT INTO "Accounts" VALUES(5,3,'Checking',1500.0);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber TEXT,
    JoinDate DATETIME DEFAULT CURRENT_TIMESTAMP);
INSERT INTO "Customers" VALUES(1,'John','Smith','john@email.com','+1234567890','2024-03-13 21:27:12');
INSERT INTO "Customers" VALUES(2,'Jane','Doe','jane@email.com','+0987654321','2024-03-13 21:27:12');
INSERT INTO "Customers" VALUES(3,'Mike','Johnson','mike@email.com','+9876543210','2024-03-13 21:27:12');
INSERT INTO "Customers" VALUES(4,'Emily','Williams','emily@email.com','+1357924680','2024-03-13 21:27:12');
INSERT INTO "Customers" VALUES(5,'David','Brown','david@email.com','+2468135790','2024-03-13 21:27:12');
CREATE TABLE InternetBankingPlatforms(
    PlatformID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT,
    AdditionalFeatures TEXT,
    FOREIGN KEY(AdditionalFeatures) REFERENCES SignInOptions(OptionName));
INSERT INTO "InternetBankingPlatforms" VALUES(1,'PlatA','Description of Plat A','Biometric Authentication');
INSERT INTO "InternetBankingPlatforms" VALUES(2,'PlatB','Description of Plat B','Password Protection');
INSERT INTO "InternetBankingPlatforms" VALUES(3,'PlatC','Description of Plat C','Token Based Authentication');
INSERT INTO "InternetBankingPlatforms" VALUES(4,'PlatD','Description of Plat D','Fingerprint Scanning');
INSERT INTO "InternetBankingPlatforms" VALUES(5,'PlatE','Description of Plat E','PIN Code');
CREATE TABLE MobileBankingApps(
    AppID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT,
    PlatformType TEXT,
    FOREIGN KEY(PlatformType) REFERENCES SignInOptions(OptionName));
INSERT INTO "MobileBankingApps" VALUES(1,'AppA','Description of App A','Biometric Authentication');
INSERT INTO "MobileBankingApps" VALUES(2,'AppB','Description of App B','Password Protection');
INSERT INTO "MobileBankingApps" VALUES(3,'AppC','Description of App C','Token Based Authentication');
INSERT INTO "MobileBankingApps" VALUES(4,'AppD','Description of App D','Fingerprint Scanning');
INSERT INTO "MobileBankingApps" VALUES(5,'AppE','Description of App E','PIN Code');
CREATE TABLE SignInOptions(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    OptionName TEXT UNIQUE);
INSERT INTO "SignInOptions" VALUES(1,'Biometric Authentication');
INSERT INTO "SignInOptions" VALUES(2,'Password Protection');
INSERT INTO "SignInOptions" VALUES(3,'Token Based Authentication');
INSERT INTO "SignInOptions" VALUES(4,'Fingerprint Scanning');
INSERT INTO "SignInOptions" VALUES(5,'PIN Code');
CREATE TABLE Transactions(
    TransactionId INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountId INTEGER,
    Amount REAL ,
    DateAndTime DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(AccountId) REFERENCES Accounts(AccountId));
INSERT INTO "Transactions" VALUES(1,1,-200.0,'2023-04-01 10:00:00');
INSERT INTO "Transactions" VALUES(2,1,50.0,'2023-04-01 11:00:00');
INSERT INTO "Transactions" VALUES(3,2,-300.0,'2023-04-01 12:00:00');
INSERT INTO "Transactions" VALUES(4,2,100.0,'2023-04-01 13:00:00');
INSERT INTO "Transactions" VALUES(5,3,-150.0,'2023-04-01 14:00:00');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('SignInOptions',5);
INSERT INTO "sqlite_sequence" VALUES('MobileBankingApps',5);
INSERT INTO "sqlite_sequence" VALUES('InternetBankingPlatforms',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
COMMIT;
