BEGIN TRANSACTION;
CREATE TABLE Article (
    articleID INTEGER PRIMARY KEY,
    headline TEXT,
    content TEXT,
    publishedDate DATE,
    journalistID INTEGER,
    FOREIGN KEY(journalistID) REFERENCES Investigative_Journalist(journalistID)
);
INSERT INTO "Article" VALUES(1,'Political Scandals Exposed','A detailed investigation into political scandals.','2021-11-01',1);
INSERT INTO "Article" VALUES(2,'Environmental Disasters Averted','An investigative report on how environmental disasters were averted thanks to timely action.','2021-11-10',2);
INSERT INTO "Article" VALUES(3,'Healthcare System Overhauled','Exploring the changes in the healthcare system that have led to improved outcomes.','2021-11-20',3);
INSERT INTO "Article" VALUES(4,'Economic Recovery Underway','How economic recovery measures are helping businesses and individuals.','2021-11-25',4);
INSERT INTO "Article" VALUES(5,'Technology Advances Transforming Society','The impact of rapid technological advancements on society.','2021-12-01',5);
CREATE TABLE "Investigative_Journalist" (
    journalistID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    degreeID INTEGER,
    FOREIGN KEY(degreeID) REFERENCES "Qualification"(degreeID));
INSERT INTO "Investigative_Journalist" VALUES(1,'John','Doe',1);
INSERT INTO "Investigative_Journalist" VALUES(2,'Jane','Smith',2);
INSERT INTO "Investigative_Journalist" VALUES(3,'Emily','Johnson',3);
INSERT INTO "Investigative_Journalist" VALUES(4,'Lisa','Brown',1);
INSERT INTO "Investigative_Journalist" VALUES(5,'Chris','Green',2);
CREATE TABLE "Qualification" (
    degreeID INTEGER PRIMARY KEY,
    title TEXT,
    major TEXT
);
INSERT INTO "Qualification" VALUES(1,'Bachelor of Arts','Investigative Journalism');
INSERT INTO "Qualification" VALUES(2,'Master of Science','Data Journalism');
INSERT INTO "Qualification" VALUES(3,'Doctor of Philosophy','Mass Communication');
CREATE TABLE Source (
    sourceID INTEGER PRIMARY KEY,
    name TEXT,
    type TEXT,
    contactInfo TEXT,
    articleID INTEGER,
    FOREIGN KEY(articleID) REFERENCES Article(articleID)
);
INSERT INTO "Source" VALUES(1,'Government Agency','Official Reports','www.govagency.com/reports',1);
INSERT INTO "Source" VALUES(2,'Nonprofit Organization','Research Papers','www.nonprofitorg.org/papers',2);
INSERT INTO "Source" VALUES(3,'Local Business','Financial Statements','www.localbusiness.com/financials',3);
INSERT INTO "Source" VALUES(4,'International NGO','Annual Reports','www.internationalgov.org/annuals',4);
INSERT INTO "Source" VALUES(5,'Academic Institution','Studies & Surveys','www.acadinst.edu/studies',5);
COMMIT;
