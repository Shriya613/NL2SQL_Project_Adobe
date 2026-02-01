BEGIN TRANSACTION;
CREATE TABLE Category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT
);
INSERT INTO "Category" VALUES(1,'Current Affairs');
INSERT INTO "Category" VALUES(2,'Politics');
INSERT INTO "Category" VALUES(3,'History');
INSERT INTO "Category" VALUES(4,'Culture');
INSERT INTO "Category" VALUES(5,'Technology');
CREATE TABLE "Creator" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Creator" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Creator" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Creator" VALUES(3,'Tom Johnson','tomjohnson@example.com');
CREATE TABLE "Documentary" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INT,
    producer_id INTEGER,
    category_id INTEGER,
    FOREIGN KEY(producer_id) REFERENCES "Creator"(id),
    FOREIGN KEY(category_id) REFERENCES Category(id)
);
INSERT INTO "Documentary" VALUES(1,'The Rise of Fascism',2019,1,1);
INSERT INTO "Documentary" VALUES(2,'Inside North Korea',2018,2,3);
INSERT INTO "Documentary" VALUES(3,'Revolutionizing Technology',2020,3,5);
INSERT INTO "Documentary" VALUES(4,'The Art of War',2017,1,4);
INSERT INTO "Documentary" VALUES(5,'Climate Change Crisis',2021,2,2);
CREATE TABLE Journalist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_name TEXT,
    email TEXT,
    documentary_id INTEGER,
    role TEXT,
    FOREIGN KEY(documentary_id) REFERENCES Documentary(id)
);
INSERT INTO "Journalist" VALUES(1,'Michael Lee','mlee@example.com',1,'Reporter');
INSERT INTO "Journalist" VALUES(2,'Emily Davis','edavis@example.com',2,'Production Assistant');
INSERT INTO "Journalist" VALUES(3,'David Williams','dwilliams@example.com',3,'Interviewer');
INSERT INTO "Journalist" VALUES(4,'Rachel Thompson','rthompson@example.com',4,'Researcher');
INSERT INTO "Journalist" VALUES(5,'Kevin Wright','kwright@example.com',5,'Presenter');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Creator',3);
INSERT INTO "sqlite_sequence" VALUES('Category',5);
INSERT INTO "sqlite_sequence" VALUES('Journalist',5);
INSERT INTO "sqlite_sequence" VALUES('Documentary',5);
COMMIT;
