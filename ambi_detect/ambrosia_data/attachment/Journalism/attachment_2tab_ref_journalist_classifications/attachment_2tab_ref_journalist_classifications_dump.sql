BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    published_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    author_id INTEGER,
    stringer_id INTEGER,
    editor_id INTEGER,
    FOREIGN KEY (author_id) REFERENCES Authors(id),
    FOREIGN KEY (stringer_id) REFERENCES Stringers(id),
    FOREIGN KEY (editor_id) REFERENCES Editors(id)
);
INSERT INTO "Articles" VALUES(1,'Article 1 Title','Content of Article 1',NULL,1,1,1);
INSERT INTO "Articles" VALUES(2,'Article 2 Title','Content of Article 2',NULL,2,2,2);
INSERT INTO "Articles" VALUES(3,'Article 3 Title','Content of Article 3',NULL,3,3,3);
INSERT INTO "Articles" VALUES(4,'Article 4 Title','Content of Article 4',NULL,4,4,4);
INSERT INTO "Articles" VALUES(5,'Article 5 Title','Content of Article 5',NULL,5,5,5);
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'John','Doe','john@example.com');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','jane@example.com');
INSERT INTO "Authors" VALUES(3,'Alice','Brown','alice@example.com');
INSERT INTO "Authors" VALUES(4,'Bob','White','bob@example.com');
INSERT INTO "Authors" VALUES(5,'Mary','Green','mary@example.com');
CREATE TABLE Editors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE
, FullName TEXT);
INSERT INTO "Editors" VALUES(1,'petergarfield@example.com','Peter Garfield');
INSERT INTO "Editors" VALUES(2,'barbarawalters@example.com','Barbara Walters');
INSERT INTO "Editors" VALUES(3,'danielcraig@example.com','Daniel Craig');
INSERT INTO "Editors" VALUES(4,'keiraknightley@example.com','Keira Knightley');
INSERT INTO "Editors" VALUES(5,'markruffalo@example.com','Mark Ruffalo');
CREATE TABLE Stringers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    phone_number TEXT UNIQUE
, FullName TEXT);
INSERT INTO "Stringers" VALUES(1,'tomhanks@example.com','+1234567890','Tom Hanks');
INSERT INTO "Stringers" VALUES(2,'emmawatson@example.com','+0987654321','Emma Watson');
INSERT INTO "Stringers" VALUES(3,'leodicaprio@example.com','+1123456789','Leo DiCaprio');
INSERT INTO "Stringers" VALUES(4,'katewinslet@example.com','+0098765432','Kate Winslet');
INSERT INTO "Stringers" VALUES(5,'chrisevans@example.com','+1123456780','Chris Evans');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Stringers',5);
INSERT INTO "sqlite_sequence" VALUES('Editors',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
