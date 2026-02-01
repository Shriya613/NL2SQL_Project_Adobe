BEGIN TRANSACTION;
CREATE TABLE Articles (
    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    author_id INTEGER,
    magazine_id INTEGER,
    FOREIGN KEY(author_id) REFERENCES Authors(author_id),
    FOREIGN KEY(magazine_id) REFERENCES Magazines(magazine_id));
INSERT INTO "Articles" VALUES(1,'Breaking News','Latest news...',1,1);
INSERT INTO "Articles" VALUES(2,'New Tech Innovations','Discover new tech products...',2,2);
INSERT INTO "Articles" VALUES(3,'Celebrity Interviews','Exclusive interviews with stars...',3,3);
INSERT INTO "Articles" VALUES(4,'Stock Market Update','Market analysis and tips...',4,4);
INSERT INTO "Articles" VALUES(5,'Yoga for Health','Benefits and practices of yoga...',5,5);
CREATE TABLE Authors (
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    occupation TEXT,
    demographic_group_id INTEGER,
    FOREIGN KEY(demographic_group_id) REFERENCES Demographic_Groups(id));
INSERT INTO "Authors" VALUES(1,'John Doe','Journalist',1);
INSERT INTO "Authors" VALUES(2,'Jane Smith','Editor',2);
INSERT INTO "Authors" VALUES(3,'Tom Brown','Photographer',3);
INSERT INTO "Authors" VALUES(4,'Alice Johnson','Reporter',4);
INSERT INTO "Authors" VALUES(5,'Bob Williams','Columnist',5);
CREATE TABLE Demographic_Groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_name TEXT,
    population INT);
INSERT INTO "Demographic_Groups" VALUES(1,'Age Group A',100000);
INSERT INTO "Demographic_Groups" VALUES(2,'Gender Group B',200000);
INSERT INTO "Demographic_Groups" VALUES(3,'Income Level C',300000);
INSERT INTO "Demographic_Groups" VALUES(4,'Education Level D',400000);
INSERT INTO "Demographic_Groups" VALUES(5,'Occupation E',500000);
CREATE TABLE Magazines (
    magazine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    publication_date DATE,
    demographic_group_id INTEGER,
    professional_field_id INTEGER,
    FOREIGN KEY(demographic_group_id) REFERENCES Demographic_Groups(id),
    FOREIGN KEY(professional_field_id) REFERENCES Professional_Fields(field_id));
INSERT INTO "Magazines" VALUES(1,'News Weekly','2023-01-01',1,1);
INSERT INTO "Magazines" VALUES(2,'Tech Monthly','2023-01-15',2,2);
INSERT INTO "Magazines" VALUES(3,'ShowBiz Daily','2023-01-07',3,3);
INSERT INTO "Magazines" VALUES(4,'Money Matters','2023-01-10',4,4);
INSERT INTO "Magazines" VALUES(5,'Wellness Magazine','2023-01-12',5,5);
CREATE TABLE Professional_Fields (
    field_id INTEGER PRIMARY KEY AUTOINCREMENT,
    field_name TEXT);
INSERT INTO "Professional_Fields" VALUES(1,'Politics');
INSERT INTO "Professional_Fields" VALUES(2,'Technology');
INSERT INTO "Professional_Fields" VALUES(3,'Entertainment');
INSERT INTO "Professional_Fields" VALUES(4,'Business');
INSERT INTO "Professional_Fields" VALUES(5,'Health');
CREATE TABLE Readers (
    reader_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    preferred_magazine_id INTEGER,
    demographic_group_id INTEGER,
    FOREIGN KEY(preferred_magazine_id) REFERENCES Magazines(magazine_id),
    FOREIGN KEY(demographic_group_id) REFERENCES Demographic_Groups(id));
INSERT INTO "Readers" VALUES(1,'Alice Jones','alice@example.com',1,1);
INSERT INTO "Readers" VALUES(2,'David Miller','david@example.com',2,2);
INSERT INTO "Readers" VALUES(3,'Emma Taylor','emma@example.com',3,3);
INSERT INTO "Readers" VALUES(4,'George Wilson','george@example.com',4,4);
INSERT INTO "Readers" VALUES(5,'Hannah King','hannah@example.com',5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Demographic_Groups',5);
INSERT INTO "sqlite_sequence" VALUES('Professional_Fields',5);
INSERT INTO "sqlite_sequence" VALUES('Magazines',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Readers',5);
COMMIT;
