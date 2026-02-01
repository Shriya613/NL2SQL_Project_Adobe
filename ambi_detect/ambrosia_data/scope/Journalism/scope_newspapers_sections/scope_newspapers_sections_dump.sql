BEGIN TRANSACTION;
CREATE TABLE Article_Authors(
    article_id INTEGER REFERENCES Articles(id) ON DELETE CASCADE,
    author_id INTEGER REFERENCES Authors(id));
INSERT INTO "Article_Authors" VALUES(1,1);
INSERT INTO "Article_Authors" VALUES(2,2);
INSERT INTO "Article_Authors" VALUES(3,3);
INSERT INTO "Article_Authors" VALUES(4,4);
INSERT INTO "Article_Authors" VALUES(5,5);
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_title TEXT,
    article_content TEXT,
    publication_date DATETIME DEFAULT CURRENT_TIMESTAMP);
INSERT INTO "Articles" VALUES(1,'Brexit Updates','Latest news on Brexit.','2024-01-30 19:19:35');
INSERT INTO "Articles" VALUES(2,'Coronavirus Update','Latest updates on coronavirus situation around the world.','2024-01-30 19:19:35');
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_name TEXT);
INSERT INTO "Authors" VALUES(1,'John Smith');
INSERT INTO "Authors" VALUES(2,'Jane Doe');
INSERT INTO "Authors" VALUES(3,'Alice Johnson');
INSERT INTO "Authors" VALUES(4,'Bob Brown');
INSERT INTO "Authors" VALUES(5,'Charlie White');
CREATE TABLE Newspaper_Authors(
    newspaper_id INTEGER REFERENCES Newspapers(id) ON DELETE CASCADE,
    author_id INTEGER REFERENCES Authors(id));
INSERT INTO "Newspaper_Authors" VALUES(1,1);
INSERT INTO "Newspaper_Authors" VALUES(2,2);
INSERT INTO "Newspaper_Authors" VALUES(3,3);
INSERT INTO "Newspaper_Authors" VALUES(4,4);
INSERT INTO "Newspaper_Authors" VALUES(5,5);
CREATE TABLE Newspapers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    newspaper_name TEXT);
INSERT INTO "Newspapers" VALUES(1,'The Times');
INSERT INTO "Newspapers" VALUES(2,'Daily Mail');
INSERT INTO "Newspapers" VALUES(3,'Financial Times');
INSERT INTO "Newspapers" VALUES(4,'Guardian');
INSERT INTO "Newspapers" VALUES(5,'Independent');
CREATE TABLE Newspapers_Sections(
    newspaper_id INTEGER REFERENCES Newspapers(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES Sections(id) ON DELETE CASCADE);
INSERT INTO "Newspapers_Sections" VALUES(1,1);
INSERT INTO "Newspapers_Sections" VALUES(2,2);
INSERT INTO "Newspapers_Sections" VALUES(3,3);
INSERT INTO "Newspapers_Sections" VALUES(4,4);
INSERT INTO "Newspapers_Sections" VALUES(5,5);
INSERT INTO "Newspapers_Sections" VALUES(2,3);
INSERT INTO "Newspapers_Sections" VALUES(1,3);
INSERT INTO "Newspapers_Sections" VALUES(4,3);
INSERT INTO "Newspapers_Sections" VALUES(5,3);
CREATE TABLE Sections(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_name TEXT);
INSERT INTO "Sections" VALUES(1,'Local News');
INSERT INTO "Sections" VALUES(2,'National News');
INSERT INTO "Sections" VALUES(3,'World News');
INSERT INTO "Sections" VALUES(4,'Technology');
INSERT INTO "Sections" VALUES(5,'Entertainment');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Sections',5);
INSERT INTO "sqlite_sequence" VALUES('Newspapers',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',2);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
COMMIT;
