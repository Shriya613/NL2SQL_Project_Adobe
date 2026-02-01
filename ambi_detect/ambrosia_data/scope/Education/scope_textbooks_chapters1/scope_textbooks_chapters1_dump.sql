BEGIN TRANSACTION;
CREATE TABLE "Chapters" (
    chapter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_name TEXT);
INSERT INTO "Chapters" VALUES(1,'Introduction');
INSERT INTO "Chapters" VALUES(2,'Algebra');
INSERT INTO "Chapters" VALUES(3,'Mechanics');
INSERT INTO "Chapters" VALUES(4,'Organic Chemistry');
CREATE TABLE Textbook_Chapters (
    textbook_chapter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    textbook_id INTEGER,
    chapter_id INTEGER,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(textbook_id) REFERENCES Textbooks(id) ON DELETE CASCADE,
    FOREIGN KEY(chapter_id) REFERENCES Chapters(chapter_id) ON DELETE CASCADE
);
INSERT INTO "Textbook_Chapters" VALUES(1,1,1,'This is an introduction to Mathematics','2024-01-30 18:17:20','2024-01-30 18:17:20');
INSERT INTO "Textbook_Chapters" VALUES(2,1,2,'Here we discuss Algebra','2024-01-30 18:17:20','2024-01-30 18:17:20');
INSERT INTO "Textbook_Chapters" VALUES(3,2,1,'Welcome to Physics','2024-01-30 18:17:20','2024-01-30 18:17:20');
INSERT INTO "Textbook_Chapters" VALUES(4,2,3,'Learning Mechanics','2024-01-30 18:17:20','2024-01-30 18:17:20');
INSERT INTO "Textbook_Chapters" VALUES(5,3,1,'Starting with Chemistry','2024-01-30 18:17:20','2024-01-30 18:17:20');
CREATE TABLE Textbooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    author TEXT,
    publication_year INT
);
INSERT INTO "Textbooks" VALUES(1,'Mathematics Fundamentals','John Smith',2018);
INSERT INTO "Textbooks" VALUES(2,'Physics Basics','Jane Doe',2019);
INSERT INTO "Textbooks" VALUES(3,'Chemistry Essentials','Michael Johnson',2020);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Textbooks',3);
INSERT INTO "sqlite_sequence" VALUES('Textbook_Chapters',6);
INSERT INTO "sqlite_sequence" VALUES('Chapters',6);
COMMIT;
