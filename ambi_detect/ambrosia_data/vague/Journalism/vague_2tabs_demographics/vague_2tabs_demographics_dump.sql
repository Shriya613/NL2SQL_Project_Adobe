BEGIN TRANSACTION;
CREATE TABLE Age_Range (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    min_age INT,
    max_age INT
);
INSERT INTO "Age_Range" VALUES(1,18,24);
INSERT INTO "Age_Range" VALUES(2,25,39);
INSERT INTO "Age_Range" VALUES(3,40,64);
INSERT INTO "Age_Range" VALUES(4,65,79);
INSERT INTO "Age_Range" VALUES(5,80,100);
CREATE TABLE Article_Readers (
    article_id INTEGER,
    reader_id INTEGER,
    FOREIGN KEY(article_id) REFERENCES Articles(id),
    FOREIGN KEY(reader_id) REFERENCES Readers(id),
    PRIMARY KEY(article_id, reader_id)
);
INSERT INTO "Article_Readers" VALUES(1,1);
INSERT INTO "Article_Readers" VALUES(2,2);
INSERT INTO "Article_Readers" VALUES(3,3);
INSERT INTO "Article_Readers" VALUES(4,4);
INSERT INTO "Article_Readers" VALUES(5,5);
CREATE TABLE Articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publish_date DATETIME,
    author_id INTEGER,
    FOREIGN KEY(author_id) REFERENCES Authors(id)
);
INSERT INTO "Articles" VALUES(1,'Investigating Corruption','An article exposing corruption in a local government.','2023-01-01 00:00:00',1);
INSERT INTO "Articles" VALUES(2,'The Impact of Climate Change','A comprehensive report on climate change effects worldwide.','2023-01-02 00:00:00',2);
INSERT INTO "Articles" VALUES(3,'Political Turmoil in Country X','Exploring political unrest and its consequences in an international context.','2023-01-03 00:00:00',3);
INSERT INTO "Articles" VALUES(4,'Economic Recovery Strategies','Discussing various approaches to economic recovery after a recession.','2023-01-04 00:00:00',4);
INSERT INTO "Articles" VALUES(5,'Preserving Endangered Species','A piece highlighting efforts to protect endangered wildlife species.','2023-01-05 00:00:00',5);
CREATE TABLE Authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_name TEXT,
    bio TEXT
);
INSERT INTO "Authors" VALUES(1,'Author A','An accomplished journalist known for investigative reporting.');
INSERT INTO "Authors" VALUES(2,'Author B','A bestselling author who writes about social issues.');
INSERT INTO "Authors" VALUES(3,'Author C','A renowned foreign correspondent with extensive experience.');
INSERT INTO "Authors" VALUES(4,'Author D','A Pulitzer Prize-winning reporter covering politics.');
INSERT INTO "Authors" VALUES(5,'Author E','A freelance writer focusing on environmental topics.');
CREATE TABLE Readers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age_range_id INTEGER,
    socioeconomic_status_id INTEGER,
    FOREIGN KEY(age_range_id) REFERENCES Age_Range(id),
    FOREIGN KEY(socioeconomic_status_id) REFERENCES Socioeconomic_Status(id)
);
INSERT INTO "Readers" VALUES(1,1,1);
INSERT INTO "Readers" VALUES(2,2,2);
INSERT INTO "Readers" VALUES(3,3,3);
INSERT INTO "Readers" VALUES(4,4,4);
INSERT INTO "Readers" VALUES(5,5,5);
CREATE TABLE Socioeconomic_Status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT,
    description TEXT
);
INSERT INTO "Socioeconomic_Status" VALUES(1,'Low','Lower income and less educated');
INSERT INTO "Socioeconomic_Status" VALUES(2,'Middle','Middle income and middle education');
INSERT INTO "Socioeconomic_Status" VALUES(3,'High','Higher income and higher education');
INSERT INTO "Socioeconomic_Status" VALUES(4,'Upper Middle','Affluent with high education');
INSERT INTO "Socioeconomic_Status" VALUES(5,'Elite','Very affluent and highly educated');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Age_Range',5);
INSERT INTO "sqlite_sequence" VALUES('Socioeconomic_Status',5);
INSERT INTO "sqlite_sequence" VALUES('Readers',5);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
