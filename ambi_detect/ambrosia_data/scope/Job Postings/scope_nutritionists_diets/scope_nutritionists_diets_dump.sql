BEGIN TRANSACTION;
CREATE TABLE Applicants (
    applicant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number VARCHAR(15)
);
INSERT INTO "Applicants" VALUES(1,'Alice','Wilson','alice.wilson@example.com','+1234567890');
INSERT INTO "Applicants" VALUES(2,'Robert','Lee','robert.lee@example.com','+9876543210');
INSERT INTO "Applicants" VALUES(3,'Anna','Martinez','anna.martinez@example.com','+0987654321');
INSERT INTO "Applicants" VALUES(4,'Mark','Taylor','mark.taylor@example.com','+1357924680');
INSERT INTO "Applicants" VALUES(5,'Laura','Hall','laura.hall@example.com','+2468135790');
CREATE TABLE ApplicationStatus (
    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_posting_id INTEGER,
    applicant_id INTEGER,
    status TEXT ,
    applied_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_posting_id) REFERENCES JobPostings(job_posting_id),
    FOREIGN KEY (applicant_id) REFERENCES Applicants(applicant_id)
);
INSERT INTO "ApplicationStatus" VALUES(1,1,1,'Applied','2024-01-30 19:23:08');
INSERT INTO "ApplicationStatus" VALUES(2,1,2,'Interview Scheduled','2024-01-30 19:23:08');
INSERT INTO "ApplicationStatus" VALUES(3,2,3,'Rejected','2024-01-30 19:23:08');
INSERT INTO "ApplicationStatus" VALUES(4,3,4,'Offer Made','2024-01-30 19:23:08');
INSERT INTO "ApplicationStatus" VALUES(5,4,5,'On Hold','2024-01-30 19:23:08');
CREATE TABLE Diets (
    diet_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    diet_type_value TEXT 
);
INSERT INTO "Diets" VALUES(1,'Vegan');
INSERT INTO "Diets" VALUES(2,'Non-vegan');
INSERT INTO "Diets" VALUES(3,'Low Carb');
INSERT INTO "Diets" VALUES(4,'Paleo');
INSERT INTO "Diets" VALUES(5,'Gluten Free');
CREATE TABLE JobPostings (
    job_posting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    location TEXT,
    salary DECIMAL(10,2) 
);
INSERT INTO "JobPostings" VALUES(1,'Registered Dietitian','Help clients achieve their nutritional goals.','New York, NY',80000);
INSERT INTO "JobPostings" VALUES(2,'Nutrition Consultant','Advise on healthy eating habits and meal planning.','Los Angeles, CA',65000);
INSERT INTO "JobPostings" VALUES(3,'Food Scientist','Research and develop new food products.','Chicago, IL',75000);
INSERT INTO "JobPostings" VALUES(4,'Health Coach','Guide clients towards healthier lifestyles.','San Francisco, CA',60000);
INSERT INTO "JobPostings" VALUES(5,'Clinical Nutritionist','Assess and treat patients with malnutrition.','Seattle, WA',85000);
CREATE TABLE Nutritionists (
    nutritionist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT
, full_name TEXT);
INSERT INTO "Nutritionists" VALUES(1,'john.smith@example.com','John Smith');
INSERT INTO "Nutritionists" VALUES(2,'jane.doe@example.com','Jane Doe');
INSERT INTO "Nutritionists" VALUES(3,'michael.brown@example.com','Michael Brown');
INSERT INTO "Nutritionists" VALUES(4,'emily.white@example.com','Emily White');
INSERT INTO "Nutritionists" VALUES(5,'david.black@example.com','David Black');
CREATE TABLE Nutritionists_Diets (
    nutritionist_id INTEGER,
    diet_type_id INTEGER,
    FOREIGN KEY (nutritionist_id) REFERENCES Nutritionists(nutritionist_id),
    FOREIGN KEY (diet_type_id) REFERENCES Diets(diet_type_id),
    PRIMARY KEY (nutritionist_id, diet_type_id)
);
INSERT INTO "Nutritionists_Diets" VALUES(1,1);
INSERT INTO "Nutritionists_Diets" VALUES(2,1);
INSERT INTO "Nutritionists_Diets" VALUES(3,1);
INSERT INTO "Nutritionists_Diets" VALUES(4,1);
INSERT INTO "Nutritionists_Diets" VALUES(5,1);
INSERT INTO "Nutritionists_Diets" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Nutritionists',5);
INSERT INTO "sqlite_sequence" VALUES('Diets',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',5);
INSERT INTO "sqlite_sequence" VALUES('Applicants',5);
INSERT INTO "sqlite_sequence" VALUES('ApplicationStatus',5);
COMMIT;
