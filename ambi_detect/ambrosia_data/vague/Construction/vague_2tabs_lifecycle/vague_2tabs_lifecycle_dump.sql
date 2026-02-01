BEGIN TRANSACTION;
CREATE TABLE Infrastructure (
    infrastructure_id INTEGER PRIMARY KEY,
    description TEXT,
    service_life_expectancy_id INTEGER,
    renewal_schedule_id INTEGER,
    FOREIGN KEY (service_life_expectancy_id) REFERENCES Service_Life_Expectancy (service_life_expectancy_id) ON DELETE CASCADE,
    FOREIGN KEY (renewal_schedule_id) REFERENCES Renewal_Schedule (renewal_schedule_id) ON DELETE CASCADE
);
INSERT INTO "Infrastructure" VALUES(1,'Bridge Abutment',1,1);
INSERT INTO "Infrastructure" VALUES(2,'Retaining Wall',2,2);
INSERT INTO "Infrastructure" VALUES(3,'Foundation Slab',3,3);
INSERT INTO "Infrastructure" VALUES(4,'Earthwork Embankment',4,4);
INSERT INTO "Infrastructure" VALUES(5,'Drainage System',5,5);
CREATE TABLE Materials (
    material_id INTEGER PRIMARY KEY,
    name TEXT,
    cost REAL 
);
INSERT INTO "Materials" VALUES(1,'Concrete',30.0);
INSERT INTO "Materials" VALUES(2,'Steel Rebar',60.0);
INSERT INTO "Materials" VALUES(3,'Lumber',20.0);
INSERT INTO "Materials" VALUES(4,'Bricks',15.0);
INSERT INTO "Materials" VALUES(5,'Cement',45.0);
CREATE TABLE Projects (
    project_id INTEGER PRIMARY KEY,
    name TEXT,
    start_date DATE,
    end_date DATE,
    infrastructure_id INTEGER,
    FOREIGN KEY (infrastructure_id) REFERENCES Infrastructure (infrastructure_id) ON DELETE SET NULL
);
INSERT INTO "Projects" VALUES(1,'Highway Bridge Repair','2023-01-01','2023-07-31',1);
INSERT INTO "Projects" VALUES(2,'Road Paving','2023-05-01','2023-10-31',2);
INSERT INTO "Projects" VALUES(3,'Building Foundation','2023-03-01','2023-09-30',3);
INSERT INTO "Projects" VALUES(4,'Stadium Construction','2023-08-01','2024-02-28',4);
INSERT INTO "Projects" VALUES(5,'Park Improvement','2023-06-01','2023-12-31',5);
CREATE TABLE Renewal_Schedule (
    renewal_schedule_id INTEGER PRIMARY KEY,
    frequency CHAR(1)  -- Yearly, Monthly, Quarterly, Semi-Annual
);
INSERT INTO "Renewal_Schedule" VALUES(1,'Yearly');
INSERT INTO "Renewal_Schedule" VALUES(2,'Quarterly');
INSERT INTO "Renewal_Schedule" VALUES(3,'Semiannually');
INSERT INTO "Renewal_Schedule" VALUES(4,'Monthly');
INSERT INTO "Renewal_Schedule" VALUES(5,'Annually');
CREATE TABLE Service_Life_Expectancy (
    service_life_expectancy_id INTEGER PRIMARY KEY,
    min_years INTEGER ,
    max_years INTEGER 
);
INSERT INTO "Service_Life_Expectancy" VALUES(1,20,40);
INSERT INTO "Service_Life_Expectancy" VALUES(2,15,30);
INSERT INTO "Service_Life_Expectancy" VALUES(3,30,60);
INSERT INTO "Service_Life_Expectancy" VALUES(4,10,20);
INSERT INTO "Service_Life_Expectancy" VALUES(5,5,15);
CREATE TABLE Tasks (
    task_id INTEGER PRIMARY KEY,
    description TEXT,
    assigned_to_worker_id INTEGER,
    due_date DATE,
    completed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (assigned_to_worker_id) REFERENCES Workers (worker_id) ON DELETE SET NULL
);
INSERT INTO "Tasks" VALUES(1,'Repair concrete cracks',1,'2023-07-15',0);
INSERT INTO "Tasks" VALUES(2,'Install new drainage system',2,'2023-10-01',0);
INSERT INTO "Tasks" VALUES(3,'Replace steel rebar',3,'2023-09-15',0);
INSERT INTO "Tasks" VALUES(4,'Paint bridge abutments',4,'2023-11-15',0);
INSERT INTO "Tasks" VALUES(5,'Level retaining wall',5,'2023-12-15',0);
CREATE TABLE Workers (
    worker_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    skill_level INTEGER 
);
INSERT INTO "Workers" VALUES(1,'John','Smith',8);
INSERT INTO "Workers" VALUES(2,'Jane','Doe',9);
INSERT INTO "Workers" VALUES(3,'Mike','Johnson',7);
INSERT INTO "Workers" VALUES(4,'Emily','Williams',9);
INSERT INTO "Workers" VALUES(5,'David','Brown',8);
COMMIT;
