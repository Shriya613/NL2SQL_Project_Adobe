BEGIN TRANSACTION;
CREATE TABLE Employers(
    employerID INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    phoneNumber TEXT UNIQUE);
INSERT INTO "Employers" VALUES(1,'Company A','companyA@example.com','123456789');
INSERT INTO "Employers" VALUES(2,'Company B','companyB@example.com','987654321');
CREATE TABLE "JobPostings"(
    jobID INTEGER PRIMARY KEY,
    title TEXT,
    postingDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    employerID INTEGER,
    skills TEXT[], -- You can store multiple skills separated by commas in this text field if you don't want separate rows for each skill.
    locationID INTEGER,
    visualArtProfession TEXT, -- This should contain one value only like 'photographer', 'videographer', etc.
    pricingID INTEGER,
    FOREIGN KEY (employerID) REFERENCES Employers(employerID),
    FOREIGN KEY (locationID) REFERENCES Locations(locationID),
    FOREIGN KEY (pricingID) REFERENCES Pricing(pricingID));
INSERT INTO "JobPostings" VALUES(1,'Freelance Graphic Designer','2024-03-13 13:08:29',1,'["Adobe Photoshop"]',1,'Photographer',1);
INSERT INTO "JobPostings" VALUES(3,'Wedding Videographer','2024-03-13 13:08:29',3,'null',3,'Videographer',2);
INSERT INTO "JobPostings" VALUES(4,'Real Estate Photography','2024-03-13 13:08:29',4,'["DSLR Camera"]',4,'Photographer',1);
INSERT INTO "JobPostings" VALUES(5,'Corporate Event Videography','2024-03-13 13:08:29',5,'null',5,'Videographer',2);
CREATE TABLE Locations(
    locationID INTEGER PRIMARY KEY,
    city TEXT,
    state TEXT,
    country TEXT);
INSERT INTO "Locations" VALUES(1,'New York','NY','USA');
INSERT INTO "Locations" VALUES(2,'Los Angeles','CA','USA');
CREATE TABLE Pricing(
    pricingID INTEGER PRIMARY KEY,
    price INT );
INSERT INTO "Pricing" VALUES(1,500);
INSERT INTO "Pricing" VALUES(2,700);
COMMIT;
