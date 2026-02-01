BEGIN TRANSACTION;
CREATE TABLE Articles(
    articleID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT);
INSERT INTO "Articles" VALUES(1,'Tech Innovations in Healthcare','A look at the latest technological advancements in healthcare.');
INSERT INTO "Articles" VALUES(2,'The Future of Finance: Digital Currencies','Exploring the potential impact of digital currencies on the finance industry.');
INSERT INTO "Articles" VALUES(3,'Revolutionizing Education Through Technology','How technology is transforming the education sector.');
INSERT INTO "Articles" VALUES(4,'Online Shopping Trends in Retail','An analysis of the growing trend towards online shopping.');
INSERT INTO "Articles" VALUES(5,'The Impact of AI on Different Industries','Understanding how artificial intelligence is affecting various industries.');
CREATE TABLE Business_Reporters(
    reporterID INTEGER PRIMARY KEY AUTOINCREMENT,
    fullName TEXT);
INSERT INTO "Business_Reporters" VALUES(1,'John Doe');
INSERT INTO "Business_Reporters" VALUES(2,'Jane Smith');
INSERT INTO "Business_Reporters" VALUES(3,'Michael Brown');
INSERT INTO "Business_Reporters" VALUES(4,'Emily Johnson');
INSERT INTO "Business_Reporters" VALUES(5,'David Williams');
CREATE TABLE Business_Reporters_Articles(
    businessReporterID INTEGER,
    articleID INTEGER,
    FOREIGN KEY (businessReporterID) REFERENCES Business_Reporters(reporterID),
    FOREIGN KEY (articleID) REFERENCES Articles(articleID));
INSERT INTO "Business_Reporters_Articles" VALUES(1,1);
INSERT INTO "Business_Reporters_Articles" VALUES(2,2);
INSERT INTO "Business_Reporters_Articles" VALUES(3,3);
INSERT INTO "Business_Reporters_Articles" VALUES(4,4);
INSERT INTO "Business_Reporters_Articles" VALUES(5,5);
CREATE TABLE Business_Reporters_Industries(
    businessReporterID INTEGER,
    industryID INTEGER,
    FOREIGN KEY (businessReporterID) REFERENCES Reporters_Industries(businessReporterID),
    FOREIGN KEY (industryID) REFERENCES Reporters_Industries(industryID));
INSERT INTO "Business_Reporters_Industries" VALUES(1,1);
INSERT INTO "Business_Reporters_Industries" VALUES(2,2);
INSERT INTO "Business_Reporters_Industries" VALUES(3,3);
INSERT INTO "Business_Reporters_Industries" VALUES(4,4);
INSERT INTO "Business_Reporters_Industries" VALUES(5,5);
INSERT INTO "Business_Reporters_Industries" VALUES(2,1);
INSERT INTO "Business_Reporters_Industries" VALUES(3,1);
INSERT INTO "Business_Reporters_Industries" VALUES(4,1);
INSERT INTO "Business_Reporters_Industries" VALUES(5,1);
CREATE TABLE Industries(
    industryID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Industries" VALUES(1,'Technology');
INSERT INTO "Industries" VALUES(2,'Healthcare');
INSERT INTO "Industries" VALUES(3,'Finance');
INSERT INTO "Industries" VALUES(4,'Education');
INSERT INTO "Industries" VALUES(5,'Retail');
CREATE TABLE Reporters_Industries(
    businessReporterID INTEGER,
    industryID INTEGER,
    FOREIGN KEY (businessReporterID) REFERENCES Business_Reporters(reporterID),
    FOREIGN KEY (industryID) REFERENCES Industries(industryID));
INSERT INTO "Reporters_Industries" VALUES(1,1);
INSERT INTO "Reporters_Industries" VALUES(2,2);
INSERT INTO "Reporters_Industries" VALUES(3,3);
INSERT INTO "Reporters_Industries" VALUES(4,4);
INSERT INTO "Reporters_Industries" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Industries',5);
INSERT INTO "sqlite_sequence" VALUES('Business_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
