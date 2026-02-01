BEGIN TRANSACTION;
CREATE TABLE Awards(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    award_name TEXT,
    winner TEXT);
INSERT INTO "Awards" VALUES(1,'Innovator of the Year','Elon Musk');
INSERT INTO "Awards" VALUES(2,'Entrepreneur of the Year','Richard Branson');
CREATE TABLE Interviews(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    time TEXT,
    interviewee TEXT);
INSERT INTO "Interviews" VALUES(1,'2021-07-25','10:00 AM','Elon Musk');
INSERT INTO "Interviews" VALUES(2,'2021-09-10','2:00 PM','Bill Gates');
CREATE TABLE "Media_Scrums"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    date TEXT,
    location TEXT,
    person TEXT);
INSERT INTO "Media_Scrums" VALUES(1,'Climate Change Summit','2021-06-30','Vancouver Convention Centre','Jack Layton');
INSERT INTO "Media_Scrums" VALUES(2,'Arts Funding','2021-09-20','Montreal Museum of Fine Arts','Michael Johnson');
CREATE TABLE News_Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    author TEXT);
INSERT INTO "News_Articles" VALUES(1,'The Rising Cost of Education','A deep dive into the increasing costs of education...','Emily Davis');
INSERT INTO "News_Articles" VALUES(2,'Technology Advancements in Healthcare','Exploring how new technologies are revolutionizing healthcare...','David Brown');
CREATE TABLE "Press_Conferences"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    date TEXT,
    location TEXT,
    person TEXT);
INSERT INTO "Press_Conferences" VALUES(1,'Budget Proposal','2021-07-01','Toronto City Hall','Jack Layton');
INSERT INTO "Press_Conferences" VALUES(2,'Education Reform','2021-08-15','Ottawa Parliament Building','Jane Smith');
CREATE TABLE Publications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_name TEXT,
    publication_date TEXT,
    editor TEXT);
INSERT INTO "Publications" VALUES(1,'Forbes Magazine','2021-08-01','Lisa Miller');
INSERT INTO "Publications" VALUES(2,'Time Magazine','2021-09-15','Harry Evans');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('News_Articles',2);
INSERT INTO "sqlite_sequence" VALUES('Interviews',2);
INSERT INTO "sqlite_sequence" VALUES('Awards',2);
INSERT INTO "sqlite_sequence" VALUES('Publications',2);
INSERT INTO "sqlite_sequence" VALUES('Press_Conferences',2);
INSERT INTO "sqlite_sequence" VALUES('Media_Scrums',2);
COMMIT;
