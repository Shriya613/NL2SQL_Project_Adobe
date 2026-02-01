BEGIN TRANSACTION;
CREATE TABLE Author(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Author" VALUES(1,'John Doe');
INSERT INTO "Author" VALUES(2,'Jane Smith');
INSERT INTO "Author" VALUES(3,'Michael Johnson');
INSERT INTO "Author" VALUES(4,'Emily Davis');
INSERT INTO "Author" VALUES(5,'Robert Brown');
CREATE TABLE Book(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    age_group TEXT ,
    illustrator TEXT,
    author_id INTEGER REFERENCES Author(id) ON DELETE SET NULL,
    publisher_id INTEGER REFERENCES Publisher(id),
    genre_id INTEGER REFERENCES Genre(id),
    main_character_id INTEGER REFERENCES Character(id));
INSERT INTO "Book" VALUES(1,'The Enchanted Forest','Children''s Book','Jane Smith',1,2,3,1);
INSERT INTO "Book" VALUES(2,'Adventures of Tom Sawyer','Children''s Book','Mark Twain',2,3,4,2);
INSERT INTO "Book" VALUES(3,'Gone With The Wind','Young Adult Book','Jane Smith',3,4,5,3);
INSERT INTO "Book" VALUES(4,'To Kill A Mockingbird','Young Adult Book','Harper Lee',4,5,6,4);
INSERT INTO "Book" VALUES(5,'1984','General','George Orwell',5,1,7,5);
CREATE TABLE Character(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_name TEXT UNIQUE);
INSERT INTO "Character" VALUES(1,'Alice');
INSERT INTO "Character" VALUES(2,'Bob');
INSERT INTO "Character" VALUES(3,'Charlie');
INSERT INTO "Character" VALUES(4,'David');
INSERT INTO "Character" VALUES(5,'Eve');
CREATE TABLE Genre(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genre_name TEXT UNIQUE);
INSERT INTO "Genre" VALUES(1,'Fantasy');
INSERT INTO "Genre" VALUES(2,'Romance');
INSERT INTO "Genre" VALUES(3,'Thriller');
INSERT INTO "Genre" VALUES(4,'Horror');
INSERT INTO "Genre" VALUES(5,'Historical Fiction');
CREATE TABLE Publisher(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Publisher" VALUES(1,'ABC Publishers');
INSERT INTO "Publisher" VALUES(2,'XYZ Publishers');
INSERT INTO "Publisher" VALUES(3,'PQR Publishers');
INSERT INTO "Publisher" VALUES(4,'LMN Publishers');
INSERT INTO "Publisher" VALUES(5,'STU Publishers');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Author',5);
INSERT INTO "sqlite_sequence" VALUES('Publisher',5);
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
INSERT INTO "sqlite_sequence" VALUES('Character',5);
INSERT INTO "sqlite_sequence" VALUES('Book',5);
COMMIT;
