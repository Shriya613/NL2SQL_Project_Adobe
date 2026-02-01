BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Author_ID INTEGER,
    Published_Date DATE,
    FOREIGN KEY (Author_ID) REFERENCES Authors(id));
INSERT INTO "Articles" VALUES(1,1,'2022-01-01');
INSERT INTO "Articles" VALUES(2,2,'2022-02-01');
INSERT INTO "Articles" VALUES(3,3,'2022-03-01');
INSERT INTO "Articles" VALUES(4,4,'2022-04-01');
INSERT INTO "Articles" VALUES(5,5,'2022-05-01');
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT);
INSERT INTO "Authors" VALUES(1,'John Doe');
INSERT INTO "Authors" VALUES(2,'Jane Smith');
INSERT INTO "Authors" VALUES(3,'Alice Johnson');
INSERT INTO "Authors" VALUES(4,'Bob Brown');
INSERT INTO "Authors" VALUES(5,'Charlie White');
CREATE TABLE News_Contents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    News_Type VARCHAR(255),
    Volume_ID INTEGER,
    FOREIGN KEY (Volume_ID) REFERENCES Volumes(id));
INSERT INTO "News_Contents" VALUES(1,'First Opinion','This is an opinion piece...','Opinion Piece',1);
INSERT INTO "News_Contents" VALUES(2,'Second Feature','Another feature article...','Feature Article',1);
INSERT INTO "News_Contents" VALUES(3,'Third Opinion','Yet another opinion piece...','Opinion Piece',2);
INSERT INTO "News_Contents" VALUES(4,'Fourth Feature','More on features...','Feature Article',2);
INSERT INTO "News_Contents" VALUES(5,'Mixed Bag','A mix of news types...','Miscellaneous',3);
CREATE TABLE Publications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Volume_ID INTEGER,
    Publisher_Name TEXT,
    FOREIGN KEY (Volume_ID) REFERENCES Volumes(id));
INSERT INTO "Publications" VALUES(1,1,'Newspaper A');
INSERT INTO "Publications" VALUES(2,2,'Newspaper B');
INSERT INTO "Publications" VALUES(3,3,'Newspaper C');
INSERT INTO "Publications" VALUES(4,4,'Newspaper D');
INSERT INTO "Publications" VALUES(5,5,'Newspaper E');
CREATE TABLE Volumes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Issue_Number TEXT UNIQUE );
INSERT INTO "Volumes" VALUES(1,'01');
INSERT INTO "Volumes" VALUES(2,'02');
INSERT INTO "Volumes" VALUES(3,'03');
INSERT INTO "Volumes" VALUES(4,'04');
INSERT INTO "Volumes" VALUES(5,'05');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Volumes',5);
INSERT INTO "sqlite_sequence" VALUES('Publications',5);
INSERT INTO "sqlite_sequence" VALUES('News_Contents',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
