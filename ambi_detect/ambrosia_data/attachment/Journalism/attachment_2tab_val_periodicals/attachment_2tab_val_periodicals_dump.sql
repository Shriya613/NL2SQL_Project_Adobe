BEGIN TRANSACTION;
CREATE TABLE Articles (
    Article_ID INTEGER PRIMARY KEY,
    Title TEXT,
    Publication_Date DATE,
    Content_Length INT,
    Author_Name TEXT,
    Newspaper_ID INTEGER,
    Magazine_ID INTEGER,
    FOREIGN KEY(Newspaper_ID) REFERENCES Newspapers(ID),
    FOREIGN KEY(Magazine_ID) REFERENCES Magazines(ID));
INSERT INTO "Articles" VALUES(1,'Climate Change is Real','2021-01-01',1500,'John Smith',1,NULL);
INSERT INTO "Articles" VALUES(2,'The Rise of AI Technologies','2021-02-15',1800,'Emma Johnson',NULL,2);
INSERT INTO "Articles" VALUES(3,'Exploring Mars with NASA Rovers','2021-03-20',1200,'David Brown',3,NULL);
INSERT INTO "Articles" VALUES(4,'Inside the Minds of Great Leaders','2021-04-10',1600,'Linda Lee',NULL,4);
INSERT INTO "Articles" VALUES(5,'The Impact of Social Media on Mental Health','2021-05-01',1400,'Christopher Green',5,NULL);
CREATE TABLE Authors (
    Author_ID INTEGER PRIMARY KEY,
    First_Name TEXT,
    Last_Name TEXT,
    Biography TEXT,
    Email TEXT UNIQUE);
INSERT INTO "Authors" VALUES(1,'John','Doe','An award winning journalist with over 20 years of experience.','john.doe@example.com');
INSERT INTO "Authors" VALUES(2,'Emma','Smith','A seasoned writer who has covered a wide range of topics.','emma.smith@example.com');
INSERT INTO "Authors" VALUES(3,'David','Brown','With expertise in science journalism, David brings a unique perspective to his work.','david.brown@example.com');
INSERT INTO "Authors" VALUES(4,'Linda','Lee','As an investigative reporter, Linda has uncovered many important stories throughout her career.','linda.lee@example.com');
INSERT INTO "Authors" VALUES(5,'Christopher','Green','His insightful articles have made him a respected voice in mental health reporting.','christopher.green@example.com');
CREATE TABLE Magazines (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Circulation_Numbers INTEGER DEFAULT 0,
    Issue_Frequency TEXT ,
    Publisher TEXT,
    Topic TEXT);
INSERT INTO "Magazines" VALUES(1,'National Geographic',50000,'Monthly','National Geographic Society','Travel & Adventure');
INSERT INTO "Magazines" VALUES(2,'Time',30000,'Weekly','Meredith Corporation','Current Affairs & World News');
INSERT INTO "Magazines" VALUES(3,'Scientific American',20000,'Biweekly','Springer Nature','Science & Technology');
INSERT INTO "Magazines" VALUES(4,'Wired',25000,'Monthly','Condé Nast','Technology & Culture');
INSERT INTO "Magazines" VALUES(5,'Vanity Fair',20000,'Monthly','Conde Nast','Entertainment & Lifestyle');
CREATE TABLE Newspapers (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Circulation_Numbers INTEGER DEFAULT 0,
    Founded_Year INTEGER,
    Headquarters TEXT,
    Editor_in_Chief TEXT);
INSERT INTO "Newspapers" VALUES(1,'The Guardian',20000,1959,'London, UK','Katharine Viner');
INSERT INTO "Newspapers" VALUES(2,'Le Monde',30000,1944,'Paris, France','Jacques Maurer');
INSERT INTO "Newspapers" VALUES(3,'El País',25000,1976,'Madrid, Spain','Carol Glatz');
INSERT INTO "Newspapers" VALUES(4,'The Hindu',18000,1878,'Chennai, India','N. Ram');
INSERT INTO "Newspapers" VALUES(5,'South China Morning Post',17000,1903,'Hong Kong, China','Paula Ma');
CREATE TABLE Subscriptions (
    Subscription_ID INTEGER PRIMARY KEY,
    Customer_Name TEXT,
    Address TEXT,
    Phone_Number TEXT,
    Subscribed_To TEXT, -- could be either 'newspaper' or 'magazine'
    Start_Date DATE,
    End_Date DATE);
INSERT INTO "Subscriptions" VALUES(1,'John Doe','123 Main St, Anytown USA','555-123-4567','Newspaper','2021-01-01','2021-12-31');
INSERT INTO "Subscriptions" VALUES(2,'Emma Smith','456 Elm St, Anothertown USA','555-987-6543','Magazine','2021-02-15','2021-12-31');
INSERT INTO "Subscriptions" VALUES(3,'David Brown','789 Oak St, Yet another town USA','555-135-7890','Newspaper','2021-03-20','2021-12-31');
INSERT INTO "Subscriptions" VALUES(4,'Linda Lee','123 Pine St, Somewhere USA','555-555-5555','Magazine','2021-04-10','2021-12-31');
INSERT INTO "Subscriptions" VALUES(5,'Christopher Green','456 Spruce St, Everytown USA','555-777-8888','Newspaper','2021-05-01','2021-12-31');
COMMIT;
