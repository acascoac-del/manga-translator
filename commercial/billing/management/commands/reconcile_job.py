from django.core.management.base import BaseCommand, CommandError
from billing.models import Job
from billing.services import finish_job


class Command(BaseCommand):
    help = 'Resolve ambiguous /run submission ONLY after checking the RunPod dashboard.'

    def add_arguments(self, parser):
        parser.add_argument('job_id')
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--remote-id')
        group.add_argument('--confirmed-not-running', action='store_true')

    def handle(self, *args, **options):
        job = Job.objects.get(pk=options['job_id'])
        if job.state != 'dispatch_unknown':
            raise CommandError('Only dispatch_unknown jobs can be reconciled here')
        if options['remote_id']:
            job.remote_id = options['remote_id']
            job.state = 'submitted'
            job.save()
        else:
            finish_job(job.pk, False, 'Envío cancelado tras revisión. Crédito devuelto.')
        self.stdout.write('Reconciled')
