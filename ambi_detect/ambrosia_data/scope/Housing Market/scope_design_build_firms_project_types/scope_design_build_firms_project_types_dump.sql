BEGIN TRANSACTION;
CREATE TABLE Design_Build_Firms(
    FirmID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT);
INSERT INTO "Design_Build_Firms" VALUES(1,'ABC Construction','123 Main St');
INSERT INTO "Design_Build_Firms" VALUES(2,'XYZ Development','456 Elm St');
INSERT INTO "Design_Build_Firms" VALUES(3,'PQR Builders','789 Oak St');
INSERT INTO "Design_Build_Firms" VALUES(4,'MNO Homes','000 Maple Ave');
INSERT INTO "Design_Build_Firms" VALUES(5,'LMN Contractors','321 Pine St');
CREATE TABLE Design_Build_Firms_Project_Types(
    FirmID INTEGER,
    TypeID INTEGER,
    FOREIGN KEY(FirmID) REFERENCES Design_Build_Firms(FirmID),
    FOREIGN KEY(TypeID) REFERENCES Project_Types(TypeID));
INSERT INTO "Design_Build_Firms_Project_Types" VALUES(1,6);
INSERT INTO "Design_Build_Firms_Project_Types" VALUES(2,6);
INSERT INTO "Design_Build_Firms_Project_Types" VALUES(3,6);
INSERT INTO "Design_Build_Firms_Project_Types" VALUES(4,6);
INSERT INTO "Design_Build_Firms_Project_Types" VALUES(5,6);
INSERT INTO "Design_Build_Firms_Project_Types" VALUES(5,1);
CREATE TABLE Project_Types(
    TypeID INTEGER PRIMARY KEY,
    Description TEXT);
INSERT INTO "Project_Types" VALUES(1,'Single Family Residential');
INSERT INTO "Project_Types" VALUES(2,'Multi-Family Residential');
INSERT INTO "Project_Types" VALUES(3,'Commercial');
INSERT INTO "Project_Types" VALUES(4,'Industrial');
INSERT INTO "Project_Types" VALUES(5,'Retail');
INSERT INTO "Project_Types" VALUES(6,'Custom Home');
COMMIT;
