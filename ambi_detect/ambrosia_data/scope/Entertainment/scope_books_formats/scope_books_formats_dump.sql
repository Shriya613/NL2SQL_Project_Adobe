BEGIN TRANSACTION;
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT);
INSERT INTO "Authors" VALUES(1,'Jane','Austen');
INSERT INTO "Authors" VALUES(2,'Charles','Dickens');
INSERT INTO "Authors" VALUES(3,'Leo','Tolstoy');
INSERT INTO "Authors" VALUES(4,'Mark','Twain');
INSERT INTO "Authors" VALUES(5,'George','Orwell');
CREATE TABLE Book_Formats(
    book_id INTEGER REFERENCES Books(id),
    format_id INTEGER REFERENCES Formats(id));
INSERT INTO "Book_Formats" VALUES(1,1);
INSERT INTO "Book_Formats" VALUES(1,2);
INSERT INTO "Book_Formats" VALUES(2,1);
INSERT INTO "Book_Formats" VALUES(2,3);
INSERT INTO "Book_Formats" VALUES(3,1);
INSERT INTO "Book_Formats" VALUES(3,4);
INSERT INTO "Book_Formats" VALUES(4,1);
INSERT INTO "Book_Formats" VALUES(4,5);
INSERT INTO "Book_Formats" VALUES(5,2);
INSERT INTO "Book_Formats" VALUES(5,1);
CREATE TABLE Books(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    author TEXT);
INSERT INTO "Books" VALUES(1,'Pride and Prejudice','Jane Austen');
INSERT INTO "Books" VALUES(2,'Great Expectations','Charles Dickens');
INSERT INTO "Books" VALUES(3,'War and Peace','Leo Tolstoy');
INSERT INTO "Books" VALUES(4,'The Adventures of Huckleberry Finn','Mark Twain');
INSERT INTO "Books" VALUES(5,'1984','George Orwell');
CREATE TABLE Books_Authors(
    book_id INTEGER REFERENCES Books(id),
    author_id INTEGER REFERENCES Authors(id));
INSERT INTO "Books_Authors" VALUES(1,1);
INSERT INTO "Books_Authors" VALUES(2,2);
INSERT INTO "Books_Authors" VALUES(3,3);
INSERT INTO "Books_Authors" VALUES(4,4);
INSERT INTO "Books_Authors" VALUES(5,5);
CREATE TABLE Formats(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Formats" VALUES(1,'Paperback');
INSERT INTO "Formats" VALUES(2,'Hardcover');
INSERT INTO "Formats" VALUES(3,'Ebook');
INSERT INTO "Formats" VALUES(4,'Audiobook');
INSERT INTO "Formats" VALUES(5,'Digital');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Formats',5);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Books',5);
COMMIT;
