BEGIN TRANSACTION;
CREATE TABLE "Books"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    type_of_book TEXT ,
    literature TEXT
);
INSERT INTO "Books" VALUES(1,'The Catcher in the Rye','Novel','Fiction Book');
INSERT INTO "Books" VALUES(2,'To Kill a Mockingbird','Novel','Non-fiction Book');
INSERT INTO "Books" VALUES(3,'A Brief History of Time','Book','Non-fiction Book');
INSERT INTO "Books" VALUES(4,'Pride and Prejudice','Classic Novel','Fiction Book');
INSERT INTO "Books" VALUES(5,'Atomic Habits','Self-help Book','Non-fiction Book');
CREATE TABLE Reviews(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rating REAL ,
    book_id INTEGER,
    student_id INTEGER,
    FOREIGN KEY (book_id) REFERENCES Books(id),
    FOREIGN KEY (student_id) REFERENCES Students(id)
);
INSERT INTO "Reviews" VALUES(1,4.5,1,1);
INSERT INTO "Reviews" VALUES(2,3.6,2,2);
INSERT INTO "Reviews" VALUES(3,4.7,3,3);
INSERT INTO "Reviews" VALUES(4,5.0,4,4);
INSERT INTO "Reviews" VALUES(5,2.8,5,5);
CREATE TABLE Students(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender CHAR(1) 
);
INSERT INTO "Students" VALUES(1,'John',20,'M');
INSERT INTO "Students" VALUES(2,'Jane',22,'W');
INSERT INTO "Students" VALUES(3,'Bob',18,'M');
INSERT INTO "Students" VALUES(4,'Alice',21,'W');
INSERT INTO "Students" VALUES(5,'Tom',19,'M');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
INSERT INTO "sqlite_sequence" VALUES('Books',5);
COMMIT;
