BEGIN TRANSACTION;
CREATE TABLE Articles(
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    PublicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    AuthorID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(ID));
INSERT INTO "Articles" VALUES(1,'Article about Politics','Content of politics article...','2021-08-01 12:00:00',1);
INSERT INTO "Articles" VALUES(2,'Article about Entertainment','Content of entertainment article...','2021-08-02 12:00:00',2);
INSERT INTO "Articles" VALUES(3,'Opinion Article','Content of opinion article...','2021-08-03 12:00:00',3);
CREATE TABLE Authors(
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Role TEXT,
    AffiliatedMediaFranchiseID INTEGER,
    FOREIGN KEY (AffiliatedMediaFranchiseID) REFERENCES MediaFranchises(ID));
INSERT INTO "Authors" VALUES(1,'John','Doe','Editorial Writer',1);
INSERT INTO "Authors" VALUES(2,'Jane','Smith','Reporter',2);
INSERT INTO "Authors" VALUES(3,'Michael','Johnson','Opinion Columnist',3);
CREATE TABLE CrossPlatformExpansions(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Business_Strategy TEXT,
    Audience TEXT,
    Platform TEXT,
    Year INT);
INSERT INTO "CrossPlatformExpansions" VALUES(1,'HBO Max','Cross-Platform Expansion','General','Streaming',2020);
INSERT INTO "CrossPlatformExpansions" VALUES(2,'Disney+','Cross-Platform Expansion','Family','Streaming',2019);
INSERT INTO "CrossPlatformExpansions" VALUES(3,'Netflix Original Series','Cross-Platform Expansion','Various','Streaming',2013);
CREATE TABLE MediaFranchises(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Business_Strategy TEXT,
    Audience TEXT,
    OtherColumn1 TEXT,
    OtherColumn2 TEXT);
INSERT INTO "MediaFranchises" VALUES(1,'New York Times','Media Franchise','Liberals','Print','Digital');
INSERT INTO "MediaFranchises" VALUES(2,'Breitbart News Network','Media Franchise','Conservatives','Website','Multimedia');
INSERT INTO "MediaFranchises" VALUES(3,'The Huffington Post','Cross-Platform Expansion','Liberals','News Aggregator','Online Magazine');
CREATE TABLE Readers(
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    AgeGroup TEXT,
    PreferredGenre TEXT,
    FavouriteMediaFranchiseID INTEGER,
    FOREIGN KEY (FavouriteMediaFranchiseID) REFERENCES MediaFranchises(ID));
INSERT INTO "Readers" VALUES(1,'Alice','Williams','Young Adult','Politics',1);
INSERT INTO "Readers" VALUES(2,'Bob','Brown','Adult','Entertainment',2);
INSERT INTO "Readers" VALUES(3,'Carol','Davis','Older Adult','Literature',3);
COMMIT;
