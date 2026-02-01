BEGIN TRANSACTION;
CREATE TABLE Article (
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    authorId INTEGER,
    pressConferenceId INTEGER,
    mediaScrumId INTEGER,
    publicationDate DATE,
    FOREIGN KEY(authorId) REFERENCES Person(id),
    FOREIGN KEY(pressConferenceId) REFERENCES Press_Conference(id),
    FOREIGN KEY(mediaScrumId) REFERENCES Media_Scrum(id)
);
INSERT INTO "Article" VALUES(1,'First article','This is my first article.',1,1,NULL,'2021-12-31');
INSERT INTO "Article" VALUES(2,'Second article','Here is another interesting piece.',2,2,NULL,'2021-12-30');
INSERT INTO "Article" VALUES(3,'Third article','Check out this exciting news!',3,NULL,1,'2021-12-30');
INSERT INTO "Article" VALUES(4,'Fourth article','Read all about it...',4,NULL,2,'2021-12-30');
INSERT INTO "Article" VALUES(5,'Fifth article','A summary of recent events.',5,NULL,1,'2021-12-30');
CREATE TABLE Media_Scrum (
    id INTEGER PRIMARY KEY,
    personId INTEGER,
    date DATE,
    time TIME,
    location TEXT,
    topic TEXT,
    FOREIGN KEY(personId) REFERENCES Person(id)
);
INSERT INTO "Media_Scrum" VALUES(1,1,'2021-12-29','18:00:00','Washington DC','Upcoming political event');
INSERT INTO "Media_Scrum" VALUES(2,2,'2021-12-28','19:00:00','Chicago','Her favorite movie of the year');
CREATE TABLE Person (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Person" VALUES(1,'John Doe');
INSERT INTO "Person" VALUES(2,'Jane Smith');
INSERT INTO "Person" VALUES(3,'Emma Watson');
INSERT INTO "Person" VALUES(4,'Tom Hanks');
INSERT INTO "Person" VALUES(5,'Oprah Winfrey');
CREATE TABLE Press_Conference (
    id INTEGER PRIMARY KEY,
    personId INTEGER,
    date DATE,
    time TIME,
    venue TEXT,
    notes TEXT,
    FOREIGN KEY(personId) REFERENCES Person(id)
);
INSERT INTO "Press_Conference" VALUES(1,1,'2021-12-31','16:00:00','New York City','Discussed new policy changes.');
INSERT INTO "Press_Conference" VALUES(2,2,'2021-12-30','17:00:00','Los Angeles','Talked about her latest book release.');
COMMIT;
