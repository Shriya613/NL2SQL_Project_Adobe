BEGIN TRANSACTION;
CREATE TABLE "Advertising"(
    ID INTEGER PRIMARY KEY AUTOINCREMENT, Magazine TEXT,
    Advertising_Revenue REAL ,
    Subscribers INT,
    Country TEXT, `Type` TEXT);
INSERT INTO "Advertising" VALUES(1,'Good Housekeeping',1000000.0,30000,'USA','Social Media Ads');
INSERT INTO "Advertising" VALUES(2,'Time',800000.0,25000,'UK','Flyers');
CREATE TABLE Articles(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    AuthorID INTEGER,
    PublishDate Date,
    FOREIGN KEY(AuthorID) REFERENCES Authors(ID));
INSERT INTO "Articles" VALUES(1,'The Rise of AI','...',1,'2021-01-01');
INSERT INTO "Articles" VALUES(2,'Climate Change Effects','...',2,'2021-02-02');
CREATE TABLE Authors(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Email TEXT,
    Birthdate Date,
    Nationality Text);
INSERT INTO "Authors" VALUES(1,'John Doe','john@example.com','1990-01-01','USA');
INSERT INTO "Authors" VALUES(2,'Jane Smith','jane@example.com','1985-02-02','Canada');
CREATE TABLE Issues(
    IssueNumber INT PRIMARY KEY,
    MagazineID INTEGER,
    ReleaseDate Date,
    FOREIGN KEY(MagazineID) REFERENCES Magazines(ID));
INSERT INTO "Issues" VALUES(1,1,'2021-01-01');
INSERT INTO "Issues" VALUES(2,1,'2021-02-01');
CREATE TABLE Magazines(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Category TEXT,
    EditorInChief TEXT,
    PublishingHouse TEXT);
INSERT INTO "Magazines" VALUES(1,'Good Housekeeping','Lifestyle','Alice Johnson','Hearst Communications');
INSERT INTO "Magazines" VALUES(2,'Time','News & Current Affairs','Bob Williams','Time Inc.');
CREATE TABLE "Subscriptions"(
    ID INTEGER PRIMARY KEY AUTOINCREMENT, Magazine TEXT,
    Subscription_Fees REAL ,
    NumberOfSubscribers INT,
    "Type" TEXT);
INSERT INTO "Subscriptions" VALUES(1,'Good Housekeeping',9.99,27000,'Monthly');
INSERT INTO "Subscriptions" VALUES(2,'National Geographic',6.99,20000,'Bi-monthly');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',2);
INSERT INTO "sqlite_sequence" VALUES('Articles',2);
INSERT INTO "sqlite_sequence" VALUES('Magazines',2);
INSERT INTO "sqlite_sequence" VALUES('Advertising',2);
INSERT INTO "sqlite_sequence" VALUES('Subscriptions',2);
COMMIT;
