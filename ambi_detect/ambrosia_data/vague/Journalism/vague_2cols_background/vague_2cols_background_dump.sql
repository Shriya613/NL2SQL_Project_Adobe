BEGIN TRANSACTION;
CREATE TABLE Article (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    author_id INTEGER,
    publication_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(author_id) REFERENCES Author(id));
INSERT INTO "Article" VALUES(1,'Sample Title','Sample Content',1,'2024-02-19 14:11:16');
INSERT INTO "Article" VALUES(2,'Another Sample Title','More Sample Content',2,'2024-02-19 14:11:16');
CREATE TABLE Author (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE);
INSERT INTO "Author" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Author" VALUES(2,'Jane Smith','jane@example.com');
CREATE TABLE Journalist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    education TEXT,
    experience INTEGER,
    photojournalist BOOLEAN DEFAULT FALSE);
INSERT INTO "Journalist" VALUES(1,'Tom Brown','tom@example.com','Bachelor of Arts',5,1);
INSERT INTO "Journalist" VALUES(2,'Alice Green','alice@example.com','Master of Science',8,0);
CREATE TABLE Publication (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    publisher_id INTEGER,
    publication_name TEXT,
    start_year INTEGER,
    end_year INTEGER,
    FOREIGN KEY(publisher_id) REFERENCES Publisher(id));
INSERT INTO "Publication" VALUES(1,1,'The Daily Sun',1900,NULL);
INSERT INTO "Publication" VALUES(2,2,'The Weekly Herald',1975,NULL);
CREATE TABLE Publisher (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    contact_email TEXT UNIQUE);
INSERT INTO "Publisher" VALUES(1,'Daily News Group','123 Main St.','editor@dailynewsgroup.com');
INSERT INTO "Publisher" VALUES(2,'Weekly Chronicles','456 Elm St.','editor@weeklychronicles.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Author',2);
INSERT INTO "sqlite_sequence" VALUES('Article',2);
INSERT INTO "sqlite_sequence" VALUES('Journalist',2);
INSERT INTO "sqlite_sequence" VALUES('Publisher',2);
INSERT INTO "sqlite_sequence" VALUES('Publication',2);
COMMIT;
