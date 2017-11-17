[CmdletBinding()]
Param(
    [string(Mandatory=$true)]
    $subscriptionIds,

    [Parameter(Mandatory=$true)]
    [string]
    $resourceGroupName,

    [string]
    $location = "North Central US",
    
    [string]
    $templateFilePath = "$PSScriptRoot\keyvault-provision.json",
    
    [string]
    $templateParametersFilePath = "$PSScriptRoot\keyvault-provision.parameters.json"
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

Write-Output "Using template file at $templateFilePath."
Write-Output "Starting deployment..."
New-AzureRmResourceGroupDeployment -Name KeyVaultDeployment -ResourceGroupName $resourceGroupName -TemplateFile $templateFilePath -TemplateParameterFile $templateParametersFilePath
