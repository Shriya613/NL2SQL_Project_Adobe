BEGIN TRANSACTION;
CREATE TABLE Articles(
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    AuthorID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Persons(PersonID));
INSERT INTO "Articles" VALUES(1,'Article 1','Content of article 1...',1);
INSERT INTO "Articles" VALUES(2,'Article 2','Content of article 2...',2);
INSERT INTO "Articles" VALUES(3,'Article 3','Content of article 3...',3);
INSERT INTO "Articles" VALUES(4,'Article 4','Content of article 4...',4);
INSERT INTO "Articles" VALUES(5,'Article 5','Content of article 5...',5);
CREATE TABLE "Events"(
    EventID INTEGER PRIMARY KEY AUTOINCREMENT, "EventName" TEXT,
    EventType TEXT,
    Date DATE,
    Location TEXT,
    PersonID INTEGER,
    FOREIGN KEY(PersonID) REFERENCES Persons(PersonID));
INSERT INTO "Events" VALUES(1,'New Year Press','Press Conference','2022-01-01','New York',1);
INSERT INTO "Events" VALUES(2,'LA Launch','Press Conference','2022-02-01','Los Angeles',2);
INSERT INTO "Events" VALUES(3,'Spring Scrum','Media Scrum','2022-03-01','Chicago',1);
INSERT INTO "Events" VALUES(4,'Dallas Debate','Media Scrum','2022-04-01','Dallas',3);
INSERT INTO "Events" VALUES(5,'Capitol Interview','Interview','2022-05-01','Washington',4);
CREATE TABLE MediaScrums(
    MediaScrumID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    StartTime TIME,
    EndTime TIME,
    PersonID INTEGER,
    FOREIGN KEY(PersonID) REFERENCES Persons(PersonID));
INSERT INTO "MediaScrums" VALUES(1,'Desc 1','10:00','11:00',1);
INSERT INTO "MediaScrums" VALUES(2,'Desc 2','12:00','13:00',3);
INSERT INTO "MediaScrums" VALUES(3,'Desc 3','14:00','15:00',5);
INSERT INTO "MediaScrums" VALUES(4,'Desc 4','16:00','17:00',2);
INSERT INTO "MediaScrums" VALUES(5,'Desc 5','18:00','19:00',4);
CREATE TABLE Persons(
    PersonID INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName TEXT UNIQUE);
INSERT INTO "Persons" VALUES(1,'John Doe');
INSERT INTO "Persons" VALUES(2,'Jane Smith');
INSERT INTO "Persons" VALUES(3,'Emily Johnson');
INSERT INTO "Persons" VALUES(4,'Michael Brown');
INSERT INTO "Persons" VALUES(5,'Cathy Williams');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Persons',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('MediaScrums',5);
INSERT INTO "sqlite_sequence" VALUES('Events',5);
COMMIT;
