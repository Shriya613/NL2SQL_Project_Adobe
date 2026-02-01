BEGIN TRANSACTION;
CREATE TABLE Authors(
    authorId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Authors" VALUES(1,'John Smith');
INSERT INTO "Authors" VALUES(2,'Jane Doe');
INSERT INTO "Authors" VALUES(3,'Michael Johnson');
INSERT INTO "Authors" VALUES(4,'Emily Davis');
INSERT INTO "Authors" VALUES(5,'William Brown');
CREATE TABLE BookAuthors(
    bookId INTEGER,
    authorId INTEGER,
    FOREIGN KEY(bookId) REFERENCES Books(id),
    FOREIGN KEY(authorId) REFERENCES Authors(authorId)
);
CREATE TABLE BookPublishers(
    bookId INTEGER,
    publisherId INTEGER,
    FOREIGN KEY(bookId) REFERENCES Books(id),
    FOREIGN KEY(publisherId) REFERENCES Publishers(publisherId)
);
CREATE TABLE Chapters(
    chapterName TEXT PRIMARY KEY,
    subject VARCHAR(50)
);
INSERT INTO "Chapters" VALUES('Introduction','History');
INSERT INTO "Chapters" VALUES('Colonial America','History');
INSERT INTO "Chapters" VALUES('Renaissance','History');
INSERT INTO "Chapters" VALUES('Industrial Revolution','History');
INSERT INTO "Chapters" VALUES('Fall of Empires','History');
CREATE TABLE Publishers(
    publisherId INTEGER PRIMARY KEY AUTOINCREMENT,
    companyName TEXT
);
INSERT INTO "Publishers" VALUES(1,'Pearson');
INSERT INTO "Publishers" VALUES(2,'McGraw-Hill');
INSERT INTO "Publishers" VALUES(3,'Cengage Learning');
INSERT INTO "Publishers" VALUES(4,'Wiley');
INSERT INTO "Publishers" VALUES(5,'Elsevier');
CREATE TABLE Textbooks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    author VARCHAR(100)
);
INSERT INTO "Textbooks" VALUES(1,'World History','John Smith');
INSERT INTO "Textbooks" VALUES(2,'American History','Jane Doe');
INSERT INTO "Textbooks" VALUES(3,'European History','Michael Johnson');
INSERT INTO "Textbooks" VALUES(4,'Ancient Civilizations','Emily Davis');
INSERT INTO "Textbooks" VALUES(5,'Modern World History','William Brown');
CREATE TABLE "Textbooks_Chapters"(
    textbookId INTEGER,
    chapterName TEXT,
    FOREIGN KEY(textbookId) REFERENCES Textbooks(id),
    FOREIGN KEY(chapterName) REFERENCES Chapters(chapterName)
);
INSERT INTO "Textbooks_Chapters" VALUES(1,'Introduction');
INSERT INTO "Textbooks_Chapters" VALUES(1,'Colonial America');
INSERT INTO "Textbooks_Chapters" VALUES(2,'Fall of Empires');
INSERT INTO "Textbooks_Chapters" VALUES(2,'Introduction');
INSERT INTO "Textbooks_Chapters" VALUES(3,'Introduction');
INSERT INTO "Textbooks_Chapters" VALUES(4,'Introduction');
INSERT INTO "Textbooks_Chapters" VALUES(5,'Introduction');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publishers',5);
INSERT INTO "sqlite_sequence" VALUES('Textbooks',5);
COMMIT;
