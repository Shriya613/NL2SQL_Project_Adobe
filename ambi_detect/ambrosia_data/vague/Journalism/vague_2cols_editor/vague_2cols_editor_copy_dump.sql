BEGIN TRANSACTION;
CREATE TABLE AdAssignments (
    AdAssignmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    AdID INTEGER,
    ArticleID INTEGER,
    AssignmentDate DATE,
    FOREIGN KEY(AdID) REFERENCES Advertisements(AdID),
    FOREIGN KEY(ArticleID) REFERENCES Articles(ArticleID)
);
INSERT INTO "AdAssignments" VALUES(1,1,1,'2021-01-01');
INSERT INTO "AdAssignments" VALUES(2,2,2,'2021-01-02');
INSERT INTO "AdAssignments" VALUES(3,3,3,'2021-01-03');
INSERT INTO "AdAssignments" VALUES(4,4,4,'2021-01-04');
INSERT INTO "AdAssignments" VALUES(5,5,5,'2021-01-05');
CREATE TABLE Advertisements (
    AdID INTEGER PRIMARY KEY,
    Description TEXT,
    ImgURL TEXT,
    StartDate DATE,
    EndDate DATE,
    Price DECIMAL(19,2) 
);
INSERT INTO "Advertisements" VALUES(1,'Car ad description','http://caradimageurl.com/img.jpg','2021-01-06','2021-01-12',10000);
INSERT INTO "Advertisements" VALUES(2,'Electronics ad description','http://electronicadimageurl.com/img.jpg','2021-01-13','2021-01-19',7500);
INSERT INTO "Advertisements" VALUES(3,'Clothing ad description','http://clothingadimageurl.com/img.jpg','2021-01-20','2021-01-26',5000);
INSERT INTO "Advertisements" VALUES(4,'Job ad description','http://jobadimageurl.com/img.jpg','2021-01-27','2021-01-31','Free');
INSERT INTO "Advertisements" VALUES(5,'Real Estate ad description','http://realestateadimageurl.com/img.jpg','2021-02-01','2021-02-05',8000);
CREATE TABLE "Articles" (
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    PublishDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    Editor TEXT,
    Final_Draft TEXT
);
INSERT INTO "Articles" VALUES(1,'First Article','This is content of first article...','2024-02-19 14:01:08','Managing Editor One','Final Draft One');
INSERT INTO "Articles" VALUES(2,'Second Article','This is content of second article...','2024-02-19 14:01:08','Managing Editor Two','Final Draft Two');
INSERT INTO "Articles" VALUES(3,'Third Article','This is content of third article...','2024-02-19 14:01:08','Managing Editor Three','Final Draft Three');
INSERT INTO "Articles" VALUES(4,'Fourth Article','This is content of fourth article...','2024-02-19 14:01:08','Managing Editor Four','Final Draft Four');
INSERT INTO "Articles" VALUES(5,'Fifth Article','This is content of fifth article...','2024-02-19 14:01:08','Managing Editor Five','Final Draft Five');
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'John','Smith','johnsmith@example.com');
INSERT INTO "Authors" VALUES(2,'Jane','Doe','janedoe@example.com');
INSERT INTO "Authors" VALUES(3,'Bob','Brown','bobbrown@example.com');
INSERT INTO "Authors" VALUES(4,'Alice','White','alicewhite@example.com');
INSERT INTO "Authors" VALUES(5,'Mike','Green','mikgreen@example.com');
CREATE TABLE Comments (
    CommentID INTEGER PRIMARY KEY AUTOINCREMENT,
    ArticleID INTEGER,
    ReaderID INTEGER,
    CommentText TEXT,
    Rating INTEGER ,
    FOREIGN KEY(ArticleID) REFERENCES Articles(ArticleID),
    FOREIGN KEY(ReaderID) REFERENCES Readers(ReaderID)
);
INSERT INTO "Comments" VALUES(1,1,1,'Great article!',5);
INSERT INTO "Comments" VALUES(2,1,2,'Interesting read.',4);
INSERT INTO "Comments" VALUES(3,2,1,'Not bad but could be better.',3);
INSERT INTO "Comments" VALUES(4,2,2,'Could use improvement.',2);
INSERT INTO "Comments" VALUES(5,3,1,'Love this!',5);
CREATE TABLE Readers (
    ReaderID INTEGER PRIMARY KEY,
    Name TEXT,
    SubscriptionDate DATE,
    PreferredEmail TEXT UNIQUE
);
INSERT INTO "Readers" VALUES(1,'Subscriber One','2021-01-01','subone@example.com');
INSERT INTO "Readers" VALUES(2,'Subscriber Two','2021-01-02','subtwo@example.com');
INSERT INTO "Readers" VALUES(3,'Subscriber Three','2021-01-03','subthree@example.com');
INSERT INTO "Readers" VALUES(4,'Subscriber Four','2021-01-04','subfour@example.com');
INSERT INTO "Readers" VALUES(5,'Subscriber Five','2021-01-05','subfive@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Comments',5);
INSERT INTO "sqlite_sequence" VALUES('AdAssignments',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
