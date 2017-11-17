[CmdletBinding()]
Param(

    [string]
    $subscriptionId,

    [Parameter(Mandatory=$true)]
    [string]
    $resourceGroupName,

	[Parameter(Mandatory=$true)]
    [string]
    $location,
    
    [string]
    $templateFilePath = "$PSScriptRoot\vm-custom-script.json"
)

$context = Get-AzureRmContext

if (!$context.Account)
{
    Write-Output "Please login first..."
    $login = Login-AzureRmAccount
}


if (!$subscriptionId)
{
    Write-Output "No subscription id given, so using default subscription from context: $($context.SubscriptionName)."
} else
{
    Write-Output "Switching context to subscription id: $subscriptionId..."
    $sub = Select-AzureRmSubscription -SubscriptionId $subscriptionId
    Write-Output "Switched to subscription: $($sub.Subscription.Name)"
}

$rg = Get-AzureRmResourceGroup -Name $resourceGroupName -ErrorAction SilentlyContinue

if($rg)
{
    Write-Output "Resource Group $($rg.ResourceGroupName) already exists...skipping creation."
} else
{
    Write-Output "Resource Group $resourceGroupName does not exist...creating $resourceGroupName."
    New-AzureRmResourceGroup -Name $resourceGroupName -Location $location
}

$vms = Get-AzureRmVM -ResourceGroupName $resourceGroupName

$gpuVMNames = foreach ($vm in $vms) `
{ 
    if ($vm.HardwareProfile.VmSize.StartsWith("Standard_N")) 
    { 
        $vm.Name 
    }
}

foreach ($vm in $vms)
{
    Write-Output "Checking $($vm.Name)"
    if ($gpuVMNames -contains $vm.Name)
    {
        Write-Output "Found GPU VM $($vm.Name)"
        $vm = Get-AzureRmVM -ResourceGroupName $resourceGroupName -Name $vm.Name

        foreach($ext in $vm.Extensions)
        {
            $ext
            Write-Output "Found extension $($ext.Name)"
            if ($ext.VirtualMachineExtensionType -eq "CustomScript")
            {
                Write-Output "Removing Custom Script $($ext.Name) on $($vm.Name)."
                Remove-AzurermVMCustomScriptExtension -ResourceGroupName $resourceGroupName -VMName $vm.Name –Name $ext.Name -Force
            }
        }
    }
}

Write-Output "Using template file at $templateFilePath."
Write-Output "Starting deployment..."
New-AzureRmResourceGroupDeployment -Name CustomScriptDeployment -ResourceGroupName $resourceGroupName -vmNames $gpuVMNames -TemplateFile $templateFilePath 