BEGIN TRANSACTION;
CREATE TABLE Articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publish_date DATE,
    magazine_id INTEGER,
    author_id INTEGER,
    FOREIGN KEY(magazine_id) REFERENCES Magazines(id),
    FOREIGN KEY(author_id) REFERENCES Authors(id)
);
INSERT INTO "Articles" VALUES(1,'First Article Title','Content of First Article...','2021-06-01',1,1);
INSERT INTO "Articles" VALUES(2,'Second Article Title','Content of Second Article...','2021-06-03',1,2);
INSERT INTO "Articles" VALUES(3,'Third Article Title','Content of Third Article...','2021-06-05',2,3);
INSERT INTO "Articles" VALUES(4,'Fourth Article Title','Content of Fourth Article...','2021-06-07',2,4);
INSERT INTO "Articles" VALUES(5,'Fifth Article Title','Content of Fifth Article...','2021-06-09',2,5);
CREATE TABLE Authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);
INSERT INTO "Authors" VALUES(1,'Alex','Smith','alex@gmail.com');
INSERT INTO "Authors" VALUES(2,'Linda','Doe','linda@yahoo.com');
INSERT INTO "Authors" VALUES(3,'Robert','Brown','robert@hotmail.com');
INSERT INTO "Authors" VALUES(4,'Cathy','White','cathy@aol.com');
INSERT INTO "Authors" VALUES(5,'David','Black','david@protonmail.com');
CREATE TABLE Columns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column_name TEXT
);
INSERT INTO "Columns" VALUES(1,'Editor''s Letter');
INSERT INTO "Columns" VALUES(2,'Feature Story');
INSERT INTO "Columns" VALUES(3,'Opinion Piece');
INSERT INTO "Columns" VALUES(4,'Book Reviews');
INSERT INTO "Columns" VALUES(5,'Travel Guide');
CREATE TABLE Magazine_Authors (
    magazine_id INTEGER,
    author_id INTEGER,
    FOREIGN KEY(magazine_id) REFERENCES Magazines(id),
    FOREIGN KEY(author_id) REFERENCES Authors(id),
    PRIMARY KEY(magazine_id, author_id)
);
INSERT INTO "Magazine_Authors" VALUES(1,1);
INSERT INTO "Magazine_Authors" VALUES(1,2);
INSERT INTO "Magazine_Authors" VALUES(2,3);
INSERT INTO "Magazine_Authors" VALUES(2,4);
INSERT INTO "Magazine_Authors" VALUES(3,5);
CREATE TABLE Magazines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    publication_date DATE,
    editor_name TEXT
);
INSERT INTO "Magazines" VALUES(1,'The Daily Times','2021-06-01','John Smith');
INSERT INTO "Magazines" VALUES(2,'Weekly News Digest','2021-07-01','Jane Doe');
INSERT INTO "Magazines" VALUES(3,'Monthly Business Review','2021-08-01','Tom Brown');
INSERT INTO "Magazines" VALUES(4,'Technology Insider','2021-09-01','Emily White');
INSERT INTO "Magazines" VALUES(5,'Fashion World','2021-10-01','Michael Black');
CREATE TABLE Magazines_Columns (
    magazine_id INTEGER,
    column_id INTEGER,
    issue_date DATE,
    FOREIGN KEY(magazine_id) REFERENCES Magazines(id),
    FOREIGN KEY(column_id) REFERENCES Columns(id),
    PRIMARY KEY(magazine_id, column_id, issue_date)
);
INSERT INTO "Magazines_Columns" VALUES(1,1,'2021-06-01');
INSERT INTO "Magazines_Columns" VALUES(1,2,'2021-06-01');
INSERT INTO "Magazines_Columns" VALUES(2,1,'2021-07-01');
INSERT INTO "Magazines_Columns" VALUES(2,2,'2021-07-01');
INSERT INTO "Magazines_Columns" VALUES(3,1,'2021-08-01');
INSERT INTO "Magazines_Columns" VALUES(3,2,'2021-08-01');
INSERT INTO "Magazines_Columns" VALUES(4,1,NULL);
INSERT INTO "Magazines_Columns" VALUES(5,1,NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Magazines',5);
INSERT INTO "sqlite_sequence" VALUES('Columns',5);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
