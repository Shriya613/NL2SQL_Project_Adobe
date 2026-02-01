BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER,
    publication_id INTEGER,
    article_title TEXT,
    publish_date DATE,
    FOREIGN KEY(journalist_id) REFERENCES Journalists(id),
    FOREIGN KEY(publication_id) REFERENCES Publications(id)
);
INSERT INTO "Articles" VALUES(1,1,1,'Political Crisis Unfolding','2021-01-01');
INSERT INTO "Articles" VALUES(2,2,2,'Economic Recovery Progress','2021-01-08');
INSERT INTO "Articles" VALUES(3,3,3,'Climate Change Effects','2021-01-15');
INSERT INTO "Articles" VALUES(4,4,4,'Technology Revolution','2021-01-22');
INSERT INTO "Articles" VALUES(5,5,5,'Healthcare Innovations','2021-01-29');
CREATE TABLE "Elements"(
    element_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT
,
    element_value TEXT);
INSERT INTO "Elements" VALUES(1,'Interviews');
INSERT INTO "Elements" VALUES(2,'Documents');
INSERT INTO "Elements" VALUES(3,'Photos');
INSERT INTO "Elements" VALUES(4,'Videos');
INSERT INTO "Elements" VALUES(5,'Witness Testimonies');
CREATE TABLE Investigative_Reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_name TEXT,
    start_date DATE,
    end_date DATE,
    article_id INTEGER,
    FOREIGN KEY(article_id) REFERENCES Articles(id)
);
INSERT INTO "Investigative_Reports" VALUES(1,'Corruption Scandal','2021-01-01','2021-01-31',1);
INSERT INTO "Investigative_Reports" VALUES(2,'Illegal Drug Trade','2021-01-01','2021-01-31',2);
INSERT INTO "Investigative_Reports" VALUES(3,'Tax Evasion Schemes','2021-01-01','2021-01-31',3);
INSERT INTO "Investigative_Reports" VALUES(4,'Financial Fraud','2021-01-01','2021-01-31',4);
INSERT INTO "Investigative_Reports" VALUES(5,'Cybercrime Surge','2021-01-01','2021-01-31',5);
CREATE TABLE "Investigative_Reports_Elements"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigative_report_id INTEGER,
    "element_id" INTEGER,
    description TEXT,
    FOREIGN KEY(investigative_report_id) REFERENCES Investigative_Reports(id),
    FOREIGN KEY("element_id") REFERENCES Elements(elements_id)
);
INSERT INTO "Investigative_Reports_Elements" VALUES(1,1,1,'Conducted interviews with key witnesses');
INSERT INTO "Investigative_Reports_Elements" VALUES(2,2,2,'Obtained confidential documents from a whistleblower');
INSERT INTO "Investigative_Reports_Elements" VALUES(3,3,3,'Collected photos at crime scenes');
INSERT INTO "Investigative_Reports_Elements" VALUES(4,4,4,'Recorded videos of protestors during the investigation');
INSERT INTO "Investigative_Reports_Elements" VALUES(5,5,5,'Gathered testimonies from eyewitnesses');
INSERT INTO "Investigative_Reports_Elements" VALUES(6,2,1,'Obtained confidential documents from a whistleblower');
INSERT INTO "Investigative_Reports_Elements" VALUES(7,3,1,'Collected photos at crime scenes');
INSERT INTO "Investigative_Reports_Elements" VALUES(8,4,1,'Recorded videos of protestors during the investigation');
INSERT INTO "Investigative_Reports_Elements" VALUES(9,5,1,'Gathered testimonies from eyewitnesses');
CREATE TABLE Journalists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Journalists" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Journalists" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Journalists" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Journalists" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Journalists" VALUES(5,'William Brown','william@example.com');
CREATE TABLE Publications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    publication_date DATE
);
INSERT INTO "Publications" VALUES(1,'The Daily News','2021-01-01');
INSERT INTO "Publications" VALUES(2,'The Weekly Times','2021-01-08');
INSERT INTO "Publications" VALUES(3,'News Today','2021-01-15');
INSERT INTO "Publications" VALUES(4,'World Report','2021-01-22');
INSERT INTO "Publications" VALUES(5,'Global Insight','2021-01-29');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Journalists',5);
INSERT INTO "sqlite_sequence" VALUES('Publications',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Investigative_Reports',5);
INSERT INTO "sqlite_sequence" VALUES('Elements',5);
INSERT INTO "sqlite_sequence" VALUES('Investigative_Reports_Elements',9);
COMMIT;
