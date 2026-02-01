BEGIN TRANSACTION;
CREATE TABLE Jobs(
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    Salary REAL,
    PostedDate DATE,
    ClosingDate DATE,
    Status TEXT ,
    MediaTypeID INTEGER,
    LanguageProficiencyID INTEGER,
    FOREIGN KEY(MediaTypeID) REFERENCES Media_Types(ID),
    FOREIGN KEY(LanguageProficiencyID) REFERENCES Languages(ID)
);
INSERT INTO "Jobs" VALUES(1,'Journalist','Write news articles for a newspaper.',60000.0,'2021-09-01','2021-09-30','Open',1,1);
INSERT INTO "Jobs" VALUES(2,'Journalist','Create content for an online magazine.',70000.0,'2021-08-15','2021-09-15','Open',2,1);
INSERT INTO "Jobs" VALUES(3,'Editor','Edit and proofread articles for a print publication.',75000.0,'2021-09-10','2021-09-25','Open',1,1);
INSERT INTO "Jobs" VALUES(4,'Editor','Review and edit content for an ebook publisher.',80000.0,'2021-08-20','2021-09-20','Open',4,2);
INSERT INTO "Jobs" VALUES(5,'Graphic Designer','Design layouts and visual elements for magazines.',65000.0,'2021-09-05','2021-09-25','Open',2,3);
INSERT INTO "Jobs" VALUES(6,'Web Developer','Develop websites and manage online presence.',85000.0,'2021-08-25','2021-09-30','Open',3,1);
INSERT INTO "Jobs" VALUES(7,'Data Analyst','Analyze and interpret data for various industries.',90000.0,'2021-09-10','2021-09-25','Closed',5,1);
CREATE TABLE Jobs_Values(
    ID INTEGER PRIMARY KEY,
    JobID INTEGER,
    ValueID INTEGER,
    FOREIGN KEY(JobID) REFERENCES Jobs(ID),
    FOREIGN KEY(ValueID) REFERENCES Media_Types_Values(ID)
);
INSERT INTO "Jobs_Values" VALUES(1,1,1);
INSERT INTO "Jobs_Values" VALUES(2,1,2);
INSERT INTO "Jobs_Values" VALUES(3,3,1);
INSERT INTO "Jobs_Values" VALUES(4,4,4);
INSERT INTO "Jobs_Values" VALUES(5,5,5);
INSERT INTO "Jobs_Values" VALUES(6,7,3);
INSERT INTO "Jobs_Values" VALUES(7,7,5);
CREATE TABLE Languages(
    ID INTEGER PRIMARY KEY,
    Name TEXT UNIQUE
);
INSERT INTO "Languages" VALUES(1,'English');
INSERT INTO "Languages" VALUES(2,'Spanish');
INSERT INTO "Languages" VALUES(3,'French');
CREATE TABLE Media_Types(
    ID INTEGER PRIMARY KEY,
    TypeName TEXT
);
INSERT INTO "Media_Types" VALUES(1,'Print');
INSERT INTO "Media_Types" VALUES(2,'Online');
CREATE TABLE Media_Types_Values(
    ID INTEGER PRIMARY KEY,
    Value TEXT UNIQUE
);
INSERT INTO "Media_Types_Values" VALUES(1,'Newspaper');
INSERT INTO "Media_Types_Values" VALUES(2,'Magazine');
INSERT INTO "Media_Types_Values" VALUES(3,'Blog');
INSERT INTO "Media_Types_Values" VALUES(4,'Ebook');
INSERT INTO "Media_Types_Values" VALUES(5,'Podcast');
COMMIT;
