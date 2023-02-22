from aws_cdk import (
    aws_ec2 as ec2,
    aws_s3 as s3,
    aws_ecs as ecs,
    aws_rds as rds,
    aws_iam as iam,
    aws_secretsmanager as sm,
    aws_ecs_patterns as ecs_patterns,
    coren
)

class DeploymentStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__()
        # ======================================
        # ========== CFN PARAMETERS ============
        project_name_param = core.CfnParameter(scope=self, id='ProjectName', type='String')
        db_name = "mlflowdb"
        port = 3306
        username = 'master'
        bucket_name = f'{project_name_param.value_as_string}-artifacts-{core.Aws.ACCOUNT_ID}'
        container_repo_name = 'mlflow-containers'
        cluster_name = 'mlflow'
        service_name = 'mlflow'
        
        # ==========================================
        # =============== IAM ROLE =================
        role = iam.Role(scope=self, id='TASKROLE', assumed_by=iam.ServicePrincipal(service='ecs-task.amazonaws.com'))
        role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonS3FullAccess'))
        role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonECS_FullAccess'))
        
        # ==========================================
        # =============== Secret ===================
        
        db_password_secret = sm.Secret(
            scope=self,
            id='DBSECRET',
            secret_name='dbPassword',
            generate_secret_string=sm.SecretStringGenerator(password_length=20, exclude_punctuation=True)            
        )
        
        # ==========================================
        # ================ VPC =====================
        public_subnet = ec2.SubnetConfiguration(name='Public', subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=28)
        # dev-shared-public-subnet-az1
        private_subnet = ec2.SubnetConfiguration(name='Private', subnet_type = ec2.SubnetType.PRIVATE, cidr_mask=28)
        isolated_subnet = ec2.SubnetConfiguration(name='DB', subnet_type=ec2.SubnetType.ISOLATED, cidr_mask=28)
        
        """
        vpc = ec2.Vpc(
            scope = self,
            id = 'VPC',
            cidr='10.0.0.0/24',
            max_azs=2,
            nat_gateway_provider=ec2.NatProvider.gateway(),
            nat_gateways=1,
            subnet_configuration=[public_subnet, private_subnet, isolated_subnet]
        )
        """
        
        """
        stack = MyStack(
            app, "MyStack", env=Environment(account="account_id", region="region")
        )
        """
        
        vpc = ec2.Vpc.from_lookup(
            self,
            "VPC",
            vpc_id = "vpc-03076add1b1efca31" #Add Correct arguments
        )
        
        # Leave, should be fine, if not check (is not NAT gateway)
        
        # ===================================================
        # ============== S3 BUCKET ==========================
        artifact_bucket = s3.Bucket(
            scope=self,
            id='ARTIFACTBUCKET',
            bucket_name=bucket_name,
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy = core.RemovalPolicy.DESTROY
        )
        
        # =========================================================
        # ========================== DATABASE =====================
        
        # Create a security group for AWS RDS
        sg_rds = ec2.SecurityGroup(scope=self, id='SGRFS', vpc=vpc, security_group_name='sg_rds')
        sg_rds.add_ingress_rule(peer=ec2.Peer.ipv4('10.206.192.0/19'), connection=ec2.Port.tcp(port))
        
        
        database = rds.DatabaseInstance(
            scope=self,
            id='MYSQL',
            database_name=db_name,
            port=port,
            credentials=rds.Credentials.from_username(username=username, password=db_password_secret.secret_value),
            engine = rds.DatabaseInstanceEngine.mysql(version=rds.MysqlEngineVersion.VER_8_0_19),
            instance_type = ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE, ec2.InstanceSize.SMALL),
            vpc=vpc,
            security_groups = [sg_rds],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE),
            removal_policy = core.RemovalPolicy.DESTROY,
            deletion_protection=False
        )
        
        # =================================================
        # ============ FARGATE SERVICE ====================
        cluster = ecs.Cluster(scope=self, id='CLUSTER', cluster_name=cluster_name, vpc=vpc)
        
        task_definition = ecs.FargateTaskDefinition(
            scope=self,
            id='MLflow',
            task_role=role
        )
        
        container = task_definition.add_container(
            id='Container',
            image = ecs.ContainerImage.from_assest(
                directory='container',
                repository_name = container_repo_name
            ),
            environment={
                'BUCKET': f's3://{artifact_bucket.bucket_name}',
                'HOST': database.db_instance_endpoint_address,
                'PORT': str(port),
                'DATABASE': db_name,
                'USERNAME': username
            },
            secrets = {
                'PASSWORD': ecs.Secret.from_secrets_manager(db_password_secret)
            }
        )
        
        port_mapping = ecs.PortMapping(container_port=5000, host_port=5000, protocol=ecs.Protocol.TCP)
        container.add_port_mappings(port_mapping)
        
        fargate_service = ecs_patterns.NetworkLoadBalancedFargateService(
            scope=self,
            id='MLFLOW',
            service_name=service_name,
            cluster=cluster,
            task_definition=task_definition
        )
        
        # Setup security group
        fargate_service.service.connections.security_groups[0].add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection = ec2.Port.tcp(5000),
            description = 'Allow inbound from VPC for mlflow'
        )
        
        scaling = fargate_service.service.auto_scale_task_count(max_capacity=2)
        scaling.scale_on_cpu_utilization(
            id='AUTOSCALING',
            target_utilization_percent=70,
            scale_in_cooldown=core.Duration.seconds(60),
            scale_out_cooldown = core.Duration.seconds(60)
        )
        
        # ====================================================
        # ========== Outputs==================================
        core.CfnOutput(scope=self, 
                       id='LoadBalancerDNS',
                       value=fargate_service.load_balancer.load_balancer_dns_name
                       )

app = core.App()
DeploymentStack(
    app,
    "DeploymentStack",
    env = {
        'account':"601883093460",
        'region': "eu-central-1"
    }
)
app.synth()