BEGIN TRANSACTION;
CREATE TABLE Articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publish_date TIMESTAMP,
    newspaper_id INTEGER,
    writer_id INTEGER,
    FOREIGN KEY(newspaper_id) REFERENCES Newspapers(id),
    FOREIGN KEY(writer_id) REFERENCES Writers(id)
);
INSERT INTO "Articles" VALUES(1,'How Technology is Changing Our Lives','In this article...','2021-10-03 10:00:00',2,1);
INSERT INTO "Articles" VALUES(2,'The Impact of Politics on Society','This piece discusses...','2021-11-07 16:00:00',4,3);
INSERT INTO "Articles" VALUES(3,'A Look at Fashion Trends','Here we explore...','2021-12-12 09:00:00',1,4);
INSERT INTO "Articles" VALUES(4,'Life Hacks from Opinion Columnists','Some useful tips...','2021-11-22 14:00:00',3,2);
INSERT INTO "Articles" VALUES(5,'Understanding Modern Technologies','Exploring the latest tech...','2021-12-17 08:00:00',1,5);
CREATE TABLE Columns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column_name TEXT
);
INSERT INTO "Columns" VALUES(1,'Opinion');
INSERT INTO "Columns" VALUES(2,'Lifestyle');
INSERT INTO "Columns" VALUES(3,'Politics');
INSERT INTO "Columns" VALUES(4,'Technology');
INSERT INTO "Columns" VALUES(5,'Fashion');
CREATE TABLE Newspapers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    publication_date DATE
);
INSERT INTO "Newspapers" VALUES(1,'The Daily News','2021-08-01');
INSERT INTO "Newspapers" VALUES(2,'The Weekly Times','2021-09-01');
INSERT INTO "Newspapers" VALUES(3,'Daily Post','2021-10-01');
INSERT INTO "Newspapers" VALUES(4,'Weekend Review','2021-11-01');
INSERT INTO "Newspapers" VALUES(5,'Morning Briefing','2021-12-01');
CREATE TABLE Newspapers_Columns (
    newspaper_id INTEGER,
    column_id INTEGER,
    FOREIGN KEY(newspaper_id) REFERENCES Newspapers(id),
    FOREIGN KEY(column_id) REFERENCES Columns(id),
    PRIMARY KEY(newspaper_id, column_id)
);
INSERT INTO "Newspapers_Columns" VALUES(1,1);
INSERT INTO "Newspapers_Columns" VALUES(1,2);
INSERT INTO "Newspapers_Columns" VALUES(2,1);
INSERT INTO "Newspapers_Columns" VALUES(2,3);
INSERT INTO "Newspapers_Columns" VALUES(3,1);
INSERT INTO "Newspapers_Columns" VALUES(4,1);
INSERT INTO "Newspapers_Columns" VALUES(5,1);
CREATE TABLE Writers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);
INSERT INTO "Writers" VALUES(1,'John','Smith','johnsmith@example.com');
INSERT INTO "Writers" VALUES(2,'Jane','Doe','janedoe@example.com');
INSERT INTO "Writers" VALUES(3,'Tom','Brown','tom@brown.com');
INSERT INTO "Writers" VALUES(4,'Alice','Green','alicelgreen@gmail.com');
INSERT INTO "Writers" VALUES(5,'Robert','White','robertwhite@hotmail.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Newspapers',5);
INSERT INTO "sqlite_sequence" VALUES('Columns',5);
INSERT INTO "sqlite_sequence" VALUES('Writers',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
