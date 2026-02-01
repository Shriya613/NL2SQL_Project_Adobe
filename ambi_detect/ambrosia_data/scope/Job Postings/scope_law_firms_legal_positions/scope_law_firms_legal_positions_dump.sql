BEGIN TRANSACTION;
CREATE TABLE Law_Firm_Job_Postings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title TEXT,
    posted_date DATE,
    application_deadline DATE,
    description TEXT,
    salary DECIMAL(10,2)
);
INSERT INTO "Law_Firm_Job_Postings" VALUES(1,'Corporate Paralegal','2022-01-01','2022-01-15','Assist attorneys in preparing documents for corporate transactions.',55000);
INSERT INTO "Law_Firm_Job_Postings" VALUES(2,'Real Estate Associate Attorney','2022-02-01','2022-02-15','Represent clients in real estate transactions.',75000);
INSERT INTO "Law_Firm_Job_Postings" VALUES(3,'Commercial Litigation Partner','2022-03-01','2022-03-15','Lead a team in handling commercial litigation cases.',150000);
INSERT INTO "Law_Firm_Job_Postings" VALUES(4,'Tax Of Counsel','2022-04-01','2022-04-15','Advise on tax matters and represent clients before tax authorities.',100000);
INSERT INTO "Law_Firm_Job_Postings" VALUES(5,'Patent Contract Attorney','2022-05-01','2022-05-15','Prepare and review patent applications for clients.',85000);
CREATE TABLE Law_Firms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    website TEXT
);
INSERT INTO "Law_Firms" VALUES(1,'Ackerman Abramovitz','6789 Central Ave., Suite 300, Dallas, TX 75206','http://www.ackermannlawyers.com');
INSERT INTO "Law_Firms" VALUES(2,'Baker & McKenzie','1 North LaSalle Street, Chicago, IL 60601-1684','https://www.bakermckenzie.com/');
INSERT INTO "Law_Firms" VALUES(3,'Mayer Brown','122 West Washington St., Chicago, IL 60602','https://www.mayerbrown.com');
INSERT INTO "Law_Firms" VALUES(4,'Skadden, Arps, Slate, Meagher & Flom LLP','One Manhattanville Rd., Purchase, NY 10577','https://www.skadden.com');
INSERT INTO "Law_Firms" VALUES(5,'DLA Piper','Three Embarcadero Center, San Francisco, CA 94111','https://www.dlapiper.com');
CREATE TABLE Law_Firms_Legal_Positions (
    firm_id INTEGER,
    position_id INTEGER,
    FOREIGN KEY (firm_id) REFERENCES Law_Firms (id),
    FOREIGN KEY (position_id) REFERENCES Legal_Positions (id),
    PRIMARY KEY (firm_id, position_id)
);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(1,1);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(1,2);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(2,1);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(2,2);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(3,1);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(3,2);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(4,1);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(4,2);
INSERT INTO "Law_Firms_Legal_Positions" VALUES(5,1);
CREATE TABLE Legal_Positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_name TEXT,
    required_experience LEVEL 
);
INSERT INTO "Legal_Positions" VALUES(1,'Paralegal','1 year');
INSERT INTO "Legal_Positions" VALUES(2,'Associate Attorney','3 years');
INSERT INTO "Legal_Positions" VALUES(3,'Partner','5 years');
INSERT INTO "Legal_Positions" VALUES(4,'Of Counsel','7 years');
INSERT INTO "Legal_Positions" VALUES(5,'Contract Attorney','2 years');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Law_Firms',5);
INSERT INTO "sqlite_sequence" VALUES('Legal_Positions',5);
INSERT INTO "sqlite_sequence" VALUES('Law_Firm_Job_Postings',5);
COMMIT;
