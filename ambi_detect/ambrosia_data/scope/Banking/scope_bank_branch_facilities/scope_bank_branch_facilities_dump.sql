BEGIN TRANSACTION;
CREATE TABLE Bank_Branch(
    BranchID INTEGER PRIMARY KEY AUTOINCREMENT,
    BranchName TEXT,
    BranchAddress TEXT);
INSERT INTO "Bank_Branch" VALUES(1,'Branch1','Address1');
INSERT INTO "Bank_Branch" VALUES(2,'Branch2','Address2');
INSERT INTO "Bank_Branch" VALUES(3,'Branch3','Address3');
INSERT INTO "Bank_Branch" VALUES(4,'Branch4','Address4');
INSERT INTO "Bank_Branch" VALUES(5,'Branch5','Address5');
CREATE TABLE Bank_Branch_Facilities(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    BranchID INTEGER,
    FacilityID INTEGER,
    FOREIGN KEY (BranchID) REFERENCES Bank_Branch(BranchID),
    FOREIGN KEY (FacilityID) REFERENCES Facilities(FacilityID));
INSERT INTO "Bank_Branch_Facilities" VALUES(1,1,1);
INSERT INTO "Bank_Branch_Facilities" VALUES(2,1,2);
INSERT INTO "Bank_Branch_Facilities" VALUES(3,2,1);
INSERT INTO "Bank_Branch_Facilities" VALUES(4,2,3);
INSERT INTO "Bank_Branch_Facilities" VALUES(5,3,1);
INSERT INTO "Bank_Branch_Facilities" VALUES(6,3,4);
INSERT INTO "Bank_Branch_Facilities" VALUES(7,4,1);
INSERT INTO "Bank_Branch_Facilities" VALUES(8,4,5);
INSERT INTO "Bank_Branch_Facilities" VALUES(9,5,1);
INSERT INTO "Bank_Branch_Facilities" VALUES(10,5,2);
CREATE TABLE Facilities(
    FacilityID INTEGER PRIMARY KEY AUTOINCREMENT,
    FacilityType TEXT);
INSERT INTO "Facilities" VALUES(1,'Drive_thru_Window');
INSERT INTO "Facilities" VALUES(2,'ATM');
INSERT INTO "Facilities" VALUES(3,'Safe_deposit_box');
INSERT INTO "Facilities" VALUES(4,'Internet_banking');
INSERT INTO "Facilities" VALUES(5,'Mobile_app');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Bank_Branch',5);
INSERT INTO "sqlite_sequence" VALUES('Facilities',5);
INSERT INTO "sqlite_sequence" VALUES('Bank_Branch_Facilities',10);
COMMIT;
