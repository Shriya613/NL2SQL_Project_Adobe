BEGIN TRANSACTION;
CREATE TABLE Advertisements (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ArticleID INTEGER,
    AdTitle TEXT,
    AdDescription TEXT,
    AdImage BLOB,
    DisplayOrder INT,
    FOREIGN KEY(ArticleID) REFERENCES Articles(ID)
);
INSERT INTO "Advertisements" VALUES(1,1,'Eco-Friendly Products','Discover our range of eco-friendly products...','image1.jpg',1);
INSERT INTO "Advertisements" VALUES(2,2,'Summer Sale','Up to 50% off on selected items!','image2.jpg',2);
CREATE TABLE Articles (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    EditorID INTEGER,
    PublishDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(EditorID) REFERENCES Editors(ID)
);
INSERT INTO "Articles" VALUES(1,'The Impact of Climate Change on Agriculture','A detailed analysis...',1,'2024-03-13 18:05:57');
INSERT INTO "Articles" VALUES(2,'The Rise of Renewable Energy Sources','An overview of...',2,'2024-03-13 18:05:57');
CREATE TABLE Editors (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Email TEXT,
    Phone_Number TEXT,
    Payment REAL
);
INSERT INTO "Editors" VALUES(1,'Alice Johnson','alice.johnson@example.com','555-123-4567',2000.0);
INSERT INTO "Editors" VALUES(2,'Bob Brown','bob.brown@example.com','555-456-7890',3000.0);
CREATE TABLE Readerships (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ArticleID INTEGER,
    SubscriptionType TEXT ,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(ArticleID) REFERENCES Articles(ID)
);
INSERT INTO "Readerships" VALUES(1,1,'Monthly','2022-01-01','2022-01-31');
INSERT INTO "Readerships" VALUES(2,2,'Yearly','2022-01-01','2023-01-01');
CREATE TABLE Stringers (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Email TEXT,
    Phone_Number TEXT,
    Payment REAL
);
INSERT INTO "Stringers" VALUES(1,'John Doe','john.doe@example.com','123-456-7890',2000.0);
INSERT INTO "Stringers" VALUES(2,'Jane Smith','jane.smith@example.com','098-765-4321',3000.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Articles',2);
INSERT INTO "sqlite_sequence" VALUES('Readerships',2);
INSERT INTO "sqlite_sequence" VALUES('Advertisements',2);
COMMIT;
