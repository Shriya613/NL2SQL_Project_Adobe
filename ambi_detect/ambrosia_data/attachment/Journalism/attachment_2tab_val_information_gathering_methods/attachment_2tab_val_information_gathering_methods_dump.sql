BEGIN TRANSACTION;
CREATE TABLE Article_Keywords(
    article_id INTEGER,
    keyword_id INTEGER,
    FOREIGN KEY(article_id) REFERENCES Articles(id),
    FOREIGN KEY(keyword_id) REFERENCES Keywords(id));
INSERT INTO "Article_Keywords" VALUES(1,1);
INSERT INTO "Article_Keywords" VALUES(1,2);
INSERT INTO "Article_Keywords" VALUES(2,3);
INSERT INTO "Article_Keywords" VALUES(2,4);
INSERT INTO "Article_Keywords" VALUES(2,5);
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Author TEXT,
    Publication_Date DATE,
    Content TEXT);
INSERT INTO "Articles" VALUES(1,'Article Title 1','Author 1','2023-01-05','Content of Article 1');
INSERT INTO "Articles" VALUES(2,'Article Title 2','Author 2','2023-01-06','Content of Article 2');
CREATE TABLE "Interviews"(
    id INTEGER PRIMARY KEY, Title TEXT,
    Date DATE,
    Location TEXT,
    Interviewer TEXT,
    Participant_Type TEXT);
INSERT INTO "Interviews" VALUES(1,'Interview Title 1','2023-01-03','Studio A','Journalist Mike Brown','Politician');
INSERT INTO "Interviews" VALUES(2,'Interview Title 2','2023-01-04','Studio B','Journalist Susan White','Business Leader');
CREATE TABLE Keywords(
    id INTEGER PRIMARY KEY,
    Word TEXT UNIQUE);
INSERT INTO "Keywords" VALUES(1,'Keyword 1');
INSERT INTO "Keywords" VALUES(2,'Keyword 2');
INSERT INTO "Keywords" VALUES(3,'Keyword 3');
INSERT INTO "Keywords" VALUES(4,'Keyword 4');
INSERT INTO "Keywords" VALUES(5,'Keyword 5');
CREATE TABLE "Press_Conferences"(
    id INTEGER PRIMARY KEY, Title TEXT,
    Date DATE,
    Location TEXT,
    Participant_Type TEXT,
    Speaker1 TEXT,
    Speaker2 TEXT);
INSERT INTO "Press_Conferences" VALUES(1,'Conference Title 1','2023-01-01','White House','Politician','President John Doe',NULL);
INSERT INTO "Press_Conferences" VALUES(2,'Conference Title 2','2023-01-02','City Hall','Government Official','Mayor Jane Smith',NULL);
CREATE TABLE Sources(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Type TEXT,
    Contact_Info TEXT);
INSERT INTO "Sources" VALUES(1,'Source Name 1','Newspaper','source@email.com; +1234567890');
INSERT INTO "Sources" VALUES(2,'Source Name 2','Magazine','source2@email.com; +0987654321');
COMMIT;
