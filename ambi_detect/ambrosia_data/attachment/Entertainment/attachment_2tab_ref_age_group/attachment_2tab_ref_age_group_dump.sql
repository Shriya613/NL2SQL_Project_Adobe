BEGIN TRANSACTION;
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Authors" VALUES(1,'Author A');
INSERT INTO "Authors" VALUES(2,'Author B');
INSERT INTO "Authors" VALUES(3,'Author C');
INSERT INTO "Authors" VALUES(4,'Author D');
INSERT INTO "Authors" VALUES(5,'Author E');
CREATE TABLE ChildrensBooks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    authorId INTEGER,
    illustratorId INTEGER,
    FOREIGN KEY(authorId) REFERENCES Authors(id),
    FOREIGN KEY(illustratorId) REFERENCES Illustrators(id)
);
INSERT INTO "ChildrensBooks" VALUES(1,'The Lion King',1,1);
INSERT INTO "ChildrensBooks" VALUES(2,'Alice in Wonderland',2,2);
INSERT INTO "ChildrensBooks" VALUES(3,'Peter Pan',3,3);
INSERT INTO "ChildrensBooks" VALUES(4,'Cinderella',4,4);
INSERT INTO "ChildrensBooks" VALUES(5,'The Little Prince',5,5);
CREATE TABLE Illustrators(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    illustratorName TEXT
);
INSERT INTO "Illustrators" VALUES(1,'Jane Smith');
INSERT INTO "Illustrators" VALUES(2,'John Doe');
INSERT INTO "Illustrators" VALUES(3,'Mary Johnson');
INSERT INTO "Illustrators" VALUES(4,'Tom Brown');
INSERT INTO "Illustrators" VALUES(5,'Emily Davis');
CREATE TABLE YoungAdultBooks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    authorId INTEGER,
    illustratorId INTEGER,
    FOREIGN KEY(authorId) REFERENCES Authors(id),
    FOREIGN KEY(illustratorId) REFERENCES Illustrators(id)
);
INSERT INTO "YoungAdultBooks" VALUES(1,'Twilight',6,1);
INSERT INTO "YoungAdultBooks" VALUES(2,'Harry Potter and the Philosopher''s Stone',7,2);
INSERT INTO "YoungAdultBooks" VALUES(3,'Percy Jackson and the Lightning Thief',8,3);
INSERT INTO "YoungAdultBooks" VALUES(4,'The Hunger Games',9,4);
INSERT INTO "YoungAdultBooks" VALUES(5,'Divergent',10,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Illustrators',5);
INSERT INTO "sqlite_sequence" VALUES('ChildrensBooks',5);
INSERT INTO "sqlite_sequence" VALUES('YoungAdultBooks',5);
COMMIT;
