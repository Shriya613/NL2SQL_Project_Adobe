BEGIN TRANSACTION;
CREATE TABLE ArticleSources (
    ArticleSourceID INTEGER PRIMARY KEY,
    ArticleID INTEGER REFERENCES Articles(ArticleID),
    SourceID INTEGER REFERENCES Sources(SourceID),
    Quote TEXT
);
INSERT INTO "ArticleSources" VALUES(1,1,1,'Quote from Associated Press source');
INSERT INTO "ArticleSources" VALUES(2,2,2,'Quote from Reuters source');
INSERT INTO "ArticleSources" VALUES(3,3,3,'Quote from Bloomberg source');
INSERT INTO "ArticleSources" VALUES(4,4,4,'Quote from CNN source');
INSERT INTO "ArticleSources" VALUES(5,5,5,'Quote from Al Jazeera source');
CREATE TABLE Articles (
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    PublishDate DATETIME,
    AuthorID INTEGER REFERENCES Reporters(ReporterID)
);
INSERT INTO "Articles" VALUES(1,'The Importance of Investigative Journalism','Investigative journalism is essential...','2022-07-01 08:00:00',1);
INSERT INTO "Articles" VALUES(2,'Climate Change and Its Impact on Society','Climate change has far-reaching effects...','2022-07-02 10:00:00',2);
INSERT INTO "Articles" VALUES(3,'Technology Advances in News Reporting','Advances in technology have revolutionized...','2022-07-03 12:00:00',3);
INSERT INTO "Articles" VALUES(4,'Coverage of Conflicts Around the World','Journalists play a crucial role in reporting...','2022-07-04 14:00:00',4);
INSERT INTO "Articles" VALUES(5,'Freedom of the Press and its Role in Democracy','A free press is vital to maintaining...','2022-07-05 16:00:00',5);
CREATE TABLE PressConferences (
    ConferenceID INTEGER PRIMARY KEY,
    Date TEXT,
    Time TIME,
    Location TEXT,
    Subject TEXT
);
INSERT INTO "PressConferences" VALUES(1,'2022-07-06','10:00:00','Washington DC','President''s Annual Address');
INSERT INTO "PressConferences" VALUES(2,'2022-07-07','14:00:00','New York City','United Nations Summit');
INSERT INTO "PressConferences" VALUES(3,'2022-07-08','16:00:00','London','G7 Leaders Meeting');
INSERT INTO "PressConferences" VALUES(4,'2022-07-09','08:00:00','Paris','EU Climate Conference');
INSERT INTO "PressConferences" VALUES(5,'2022-07-10','12:00:00','Tokyo','Asia Pacific Economic Cooperation (APEC)');
CREATE TABLE Reporters (
    ReporterID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Reporters" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Reporters" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Reporters" VALUES(3,'Michael','Brown','michael.brown@example.com');
INSERT INTO "Reporters" VALUES(4,'Emily','White','emily.white@example.com');
INSERT INTO "Reporters" VALUES(5,'David','Green','david.green@example.com');
CREATE TABLE ReportersAttending (
    ReporterAttendeeID INTEGER PRIMARY KEY,
    ReporterID INTEGER REFERENCES Reporters(ReporterID),
    ConferenceID INTEGER REFERENCES PressConferences(ConferenceID)
);
INSERT INTO "ReportersAttending" VALUES(1,1,1);
INSERT INTO "ReportersAttending" VALUES(2,2,2);
INSERT INTO "ReportersAttending" VALUES(3,3,3);
INSERT INTO "ReportersAttending" VALUES(4,4,4);
INSERT INTO "ReportersAttending" VALUES(5,5,5);
CREATE TABLE Sources (
    SourceID INTEGER PRIMARY KEY,
    Name TEXT,
    ContactInfo TEXT
);
INSERT INTO "Sources" VALUES(1,'Associated Press','https://www.apnews.com/contact');
INSERT INTO "Sources" VALUES(2,'Reuters','https://www.reuters.com/contact');
INSERT INTO "Sources" VALUES(3,'Bloomberg','https://www.bloomberg.com/contact');
INSERT INTO "Sources" VALUES(4,'CNN','https://edition.cnn.com/contacts');
INSERT INTO "Sources" VALUES(5,'Al Jazeera','http://www.aljazeera.net/aboutus/contacts');
COMMIT;
