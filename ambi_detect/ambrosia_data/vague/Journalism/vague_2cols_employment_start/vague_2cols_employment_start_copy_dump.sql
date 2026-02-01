BEGIN TRANSACTION;
CREATE TABLE Articles (
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    PublicationDate DATE,
    AuthorID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Reporters(ID)
);
INSERT INTO "Articles" VALUES(1,'The Importance of Investigative Journalism','This article discusses...','2021-01-01',1);
INSERT INTO "Articles" VALUES(2,'Impact of Social Media on News Consumption','This article examines...','2021-01-15',2);
INSERT INTO "Articles" VALUES(3,'Climate Change: A Global Crisis','This article delves into...','2021-02-01',3);
INSERT INTO "Articles" VALUES(4,'Covering Conflicts: The Role of Journalists','This article explores...','2021-02-15',4);
INSERT INTO "Articles" VALUES(5,'Journalism Ethics: Balancing Truth and Responsibility','This article analyzes...','2021-03-01',5);
CREATE TABLE Interviews (
    InterviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    Date CONTEXT,
    SourceId INTEGER,
    Notes TEXT,
    FOREIGN KEY(SourceId) REFERENCES Sources(SourceId)
);
INSERT INTO "Interviews" VALUES(1,'2021-01-20',1,'Notes about interview with Alice Cooper');
INSERT INTO "Interviews" VALUES(2,'2021-02-10',2,'Notes about interview with Bob Davis');
INSERT INTO "Interviews" VALUES(3,'2021-02-20',3,'Notes about interview with Carol Thompson');
INSERT INTO "Interviews" VALUES(4,'2021-03-05',4,'Notes about interview with Daniel Lee');
INSERT INTO "Interviews" VALUES(5,'2021-03-15',5,'Notes about interview with Elizabeth White');
CREATE TABLE Magazines (
    MagazineId INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    EditorInChiefID INTEGER,
    FOREIGN KEY(EditorInChiefID) REFERENCES Reporters(ID)
);
INSERT INTO "Magazines" VALUES(1,'World Press',1);
INSERT INTO "Magazines" VALUES(2,'American Chronicle',2);
INSERT INTO "Magazines" VALUES(3,'European Observer',3);
INSERT INTO "Magazines" VALUES(4,'Asian Insights',4);
INSERT INTO "Magazines" VALUES(5,'African Herald',5);
CREATE TABLE "Reporters" (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Email TEXT,
    PhoneNumber TEXT,
    Employment_Start INTEGER,
    ReporterName TEXT
);
INSERT INTO "Reporters" VALUES(1,'John Doe','john@example.com','+1234567890',2005,'JD');
INSERT INTO "Reporters" VALUES(2,'Jane Smith','jane@example.com','+0987654321',2006,'JS');
INSERT INTO "Reporters" VALUES(3,'Michael Johnson','michael@example.com','+1357924680',2007,'MJ');
INSERT INTO "Reporters" VALUES(4,'Emma Wilson','emma@example.com','+2468135790',2008,'EW');
INSERT INTO "Reporters" VALUES(5,'David Brown','david@example.com','+3690123456',2009,'DB');
CREATE TABLE Sources (
    SourceId INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    ContactInfo TEXT,
    LastContacted DATE
);
INSERT INTO "Sources" VALUES(1,'Alice Cooper','alice@example.org','2021-01-10');
INSERT INTO "Sources" VALUES(2,'Bob Davis','bob@example.net','2021-02-05');
INSERT INTO "Sources" VALUES(3,'Carol Thompson','carol@example.co.uk','2021-02-15');
INSERT INTO "Sources" VALUES(4,'Daniel Lee','daniel@example.ca','2021-03-05');
INSERT INTO "Sources" VALUES(5,'Elizabeth White','elizabeth@example.au','2021-03-10');
CREATE TABLE Stories (
    StoryId INTEGER PRIMARY KEY AUTOINCREMENT,
    Headline TEXT,
    Content TEXT,
    PublishDate DATE,
    AssignedToReporterID INTEGER,
    MagazineId INTEGER,
    FOREIGN KEY(AssignedToReporterID) REFERENCES Reporters(ID),
    FOREIGN KEY(MagazineId) REFERENCES Magazines(MagazineId)
);
INSERT INTO "Stories" VALUES(1,'Breakthrough in Climate Research','This story covers a recent breakthrough...','2021-01-10',1,1);
INSERT INTO "Stories" VALUES(2,'Political Turmoil in South America','This story dives into the political situation...','2021-01-15',2,2);
INSERT INTO "Stories" VALUES(3,'Healthcare System Overhaul Proposed','This story reports on a proposed overhaul...','2021-02-01',3,3);
INSERT INTO "Stories" VALUES(4,'Revolutionary Technology in Space Exploration','This story highlights revolutionary technology...','2021-02-15',4,4);
INSERT INTO "Stories" VALUES(5,'Economic Recovery Strategies Debated','This story examines various economic recovery strategies...','2021-03-01',5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Sources',5);
INSERT INTO "sqlite_sequence" VALUES('Interviews',5);
INSERT INTO "sqlite_sequence" VALUES('Magazines',5);
INSERT INTO "sqlite_sequence" VALUES('Stories',5);
COMMIT;
