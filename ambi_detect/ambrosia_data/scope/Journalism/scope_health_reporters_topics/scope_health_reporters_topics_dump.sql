BEGIN TRANSACTION;
CREATE TABLE Articles (
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    PublishDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    ReporterID INTEGER,
    FOREIGN KEY(ReporterID) REFERENCES Health_Reporters(ReporterID)
);
INSERT INTO "Articles" VALUES(1,'Understanding Diabetes','A comprehensive article on diabetes...','2024-01-30 19:01:54',1);
INSERT INTO "Articles" VALUES(2,'The Importance of a Balanced Diet','An insightful piece about the significance of balanced diets...','2024-01-30 19:01:54',5);
INSERT INTO "Articles" VALUES(3,'Depression and Its Impact on Society','A thought-provoking exploration of depression...','2024-01-30 19:01:54',3);
INSERT INTO "Articles" VALUES(4,'Cancer Prevention Techniques','Valuable information on preventing cancer...','2024-01-30 19:01:54',4);
INSERT INTO "Articles" VALUES(5,'Vitamins and Minerals: Essential for Good Health','A detailed guide on essential vitamins and minerals...','2024-01-30 19:01:54',2);
CREATE TABLE Health_Reporters (
    ReporterID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Email TEXT
);
INSERT INTO "Health_Reporters" VALUES(1,'Alice Johnson','alice@journalist.com');
INSERT INTO "Health_Reporters" VALUES(2,'Bob Smith','bob@journalist.com');
INSERT INTO "Health_Reporters" VALUES(3,'Carol Brown','carol@journalist.com');
INSERT INTO "Health_Reporters" VALUES(4,'David Green','david@journalist.com');
INSERT INTO "Health_Reporters" VALUES(5,'Eva White','eva@journalist.com');
CREATE TABLE Health_Reporters_Topics (
    ReporterID INTEGER REFERENCES Health_Reporters(ReporterID),
    TopicID INTEGER REFERENCES Topics(TopicID),
    PrimaryKey INTEGER PRIMARY KEY AUTOINCREMENT
);
INSERT INTO "Health_Reporters_Topics" VALUES(1,4,1);
INSERT INTO "Health_Reporters_Topics" VALUES(2,4,2);
INSERT INTO "Health_Reporters_Topics" VALUES(3,4,3);
INSERT INTO "Health_Reporters_Topics" VALUES(4,4,4);
INSERT INTO "Health_Reporters_Topics" VALUES(5,1,5);
INSERT INTO "Health_Reporters_Topics" VALUES(5,4,6);
CREATE TABLE Sources (
    SourceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    URL TEXT
);
INSERT INTO "Sources" VALUES(1,'WebMD','https://www.webmd.com/');
INSERT INTO "Sources" VALUES(2,'Mayo Clinic','https://www.mayoclinic.org/');
INSERT INTO "Sources" VALUES(3,'Johns Hopkins Medicine','https://www.hopkinsmedicine.org/');
INSERT INTO "Sources" VALUES(4,'Harvard Medical School','https://www.hsph.harvard.edu/');
INSERT INTO "Sources" VALUES(5,'National Institutes of Health','https://www.nih.gov/');
CREATE TABLE Topics (
    TopicID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Topics" VALUES(1,'Nutrition');
INSERT INTO "Topics" VALUES(2,'Mental Health');
INSERT INTO "Topics" VALUES(3,'Diseases');
INSERT INTO "Topics" VALUES(4,'Prevention');
INSERT INTO "Topics" VALUES(5,'Treatments');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Health_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Topics',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Sources',5);
INSERT INTO "sqlite_sequence" VALUES('Health_Reporters_Topics',6);
COMMIT;
