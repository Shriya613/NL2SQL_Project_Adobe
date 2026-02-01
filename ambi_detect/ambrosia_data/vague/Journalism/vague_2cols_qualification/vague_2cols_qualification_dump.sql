BEGIN TRANSACTION;
CREATE TABLE articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publishDate DATETIME,
    authorId INTEGER,
    publicationId INTEGER,
    topicId INTEGER,
    FOREIGN KEY (authorId) REFERENCES journalists(id),
    FOREIGN KEY (publicationId) REFERENCES publications(id),
    FOREIGN KEY (topicId) REFERENCES topics(id)
);
INSERT INTO "articles" VALUES(1,'Breaking News: New Government Policy','The government has announced a new policy that will impact citizens.','2023-04-01T10:00:00Z',1,1,1);
INSERT INTO "articles" VALUES(2,'Economic Outlook for Q2','Experts discuss the economic outlook for the second quarter.','2023-04-02T10:00:00Z',2,2,3);
CREATE TABLE journalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phoneNumber TEXT,
    Degree TEXT,
    Certificate TEXT,
    Investigative_Journalist BOOLEAN
);
INSERT INTO "journalists" VALUES(1,'John Doe','john@example.com','123-456-7890','Bachelor of Arts in Journalism','Certified Professional Writer',1);
INSERT INTO "journalists" VALUES(2,'Jane Smith','jane@example.com','098-765-4321','Master of Science in Communication','Advanced Reporting Specialist',0);
CREATE TABLE publications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT
);
INSERT INTO "publications" VALUES(1,'Daily Times','123 Main St, Anytown, USA');
INSERT INTO "publications" VALUES(2,'Weekly Post','456 Elm St, Anycity, USA');
CREATE TABLE sources(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    contactInfo TEXT
);
INSERT INTO "sources" VALUES(1,'City Hall','Phone: +1 (555) 123-4567 | Email: cityhall@anycity.gov');
INSERT INTO "sources" VALUES(2,'Local Police Department','Phone: +1 (555) 987-6543 | Email: police@anystate.us');
CREATE TABLE topics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT
);
INSERT INTO "topics" VALUES(1,'Government and Politics');
INSERT INTO "topics" VALUES(2,'Business and Economy');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('journalists',2);
INSERT INTO "sqlite_sequence" VALUES('articles',2);
INSERT INTO "sqlite_sequence" VALUES('publications',2);
INSERT INTO "sqlite_sequence" VALUES('sources',2);
INSERT INTO "sqlite_sequence" VALUES('topics',2);
COMMIT;
