BEGIN TRANSACTION;
CREATE TABLE Article_Publication (
    article_publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    freelance_journalist_id INTEGER,
    assignment_id INTEGER,
    publication_id INTEGER,
    publish_date DATE,
    FOREIGN KEY(freelance_journalist_id) REFERENCES Freelance_Journalists(journalist_id),
    FOREIGN KEY(assignment_id) REFERENCES Assignments(assignment_id),
    FOREIGN KEY(publication_id) REFERENCES Publications(publication_id)
);
INSERT INTO "Article_Publication" VALUES(1,1,1,1,'2023-01-17');
INSERT INTO "Article_Publication" VALUES(2,1,2,2,'2023-01-22');
INSERT INTO "Article_Publication" VALUES(3,2,1,3,'2023-01-26');
INSERT INTO "Article_Publication" VALUES(4,2,3,4,'2023-02-01');
INSERT INTO "Article_Publication" VALUES(5,3,2,5,'2023-02-08');
INSERT INTO "Article_Publication" VALUES(6,3,1,5,'2023-02-08');
INSERT INTO "Article_Publication" VALUES(7,4,1,NULL,NULL);
INSERT INTO "Article_Publication" VALUES(8,5,1,NULL,NULL);
CREATE TABLE Assignments (
    assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    deadline DATE
);
INSERT INTO "Assignments" VALUES(1,'Feature Stories','2023-01-15');
INSERT INTO "Assignments" VALUES(2,'News Articles','2023-01-20');
INSERT INTO "Assignments" VALUES(3,'Investigative Reports','2023-01-25');
INSERT INTO "Assignments" VALUES(4,'Profile Pieces','2023-01-30');
INSERT INTO "Assignments" VALUES(5,'Opinion Pieces','2023-02-05');
CREATE TABLE Freelance_Journalists (
    journalist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Freelance_Journalists" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Freelance_Journalists" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Freelance_Journalists" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Freelance_Journalists" VALUES(4,'Emily Williams','emily@example.com');
INSERT INTO "Freelance_Journalists" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Publications (
    publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_name TEXT,
    publisher_name TEXT
);
INSERT INTO "Publications" VALUES(1,'The Daily News','Daily Newspaper Ltd.');
INSERT INTO "Publications" VALUES(2,'The Weekly Times','Weekly Publishing Co.');
INSERT INTO "Publications" VALUES(3,'The Monthly Magazine','Monthly Magazines Inc.');
INSERT INTO "Publications" VALUES(4,'The Annual Review','Annual Reviews Corp.');
INSERT INTO "Publications" VALUES(5,'The Quarterly Report','Quarterly Publishers LLC.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Freelance_Journalists',5);
INSERT INTO "sqlite_sequence" VALUES('Assignments',5);
INSERT INTO "sqlite_sequence" VALUES('Publications',5);
INSERT INTO "sqlite_sequence" VALUES('Article_Publication',8);
COMMIT;
